from langchain import ConversationChain
import streamlit as st
import csv
import base64
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
from langchain.llms import HuggingFacePipeline
import torch
from langchain.chains.question_answering import load_qa_chain
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from typing import Literal
from dataclasses import dataclass
# chat_placeholder = st.container()
# prompt_placeholder = st.form("chat-form")
# #Logo

# @dataclass
# class Message:
#     """Class for keeping track of a chat message."""
#     origin: Literal["human", "ai"]
#     message: str

with st.sidebar:
    # logo_path = "Logo/ltts.png"
    # st.image(logo_path, caption=None, width=200, use_column_width="auto", clamp=False, channels="RGB", output_format="auto")
    #File Uploders
    file_1 = st.file_uploader("Upload your Files", key="file_1")
    if file_1:
        file_extension = file_1.name.split(".")[-1].lower()
        mime_type = file_1.type.lower()
        if file_extension == 'pdf' or mime_type == 'application/pdf':
            pdf = file_1
            pdf_contents = pdf.read()
            st.write("PDF Preview")
            pdf_base64 = base64.b64encode(pdf_contents).decode('utf-8')
            preview = f'<embed src="data:application/pdf;base64,{pdf_base64}" width="400" height="500" type="application/pdf">'
            st.markdown(preview, unsafe_allow_html=True )
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
                text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=200,
                chunk_overlap=100,
                length_function=len
        )
            chunks = text_splitter.split_text(text=text)
            store_name = pdf.name[:-4]
            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
            else:
                embeddings = HuggingFaceEmbeddings()
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)

#     file_2 = st.file_uploader("", key="file_2")
#     file_3 = st.file_uploader("", key="file_3")
#     file_4 = st.file_uploader("", key="file_4")

##
model = 'declare-lab/flan-alpaca-base'
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForSeq2SeqLM.from_pretrained(model, 
                                                        device_map='auto',
                                                        offload_folder="save_folder",
                                                        torch_dtype=torch.float32)
pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=300,
                model_kwargs={"temperature": 0.1},
        )
llm = HuggingFacePipeline(pipeline=pipe)
##

st.title('Test Case Generation')
prompt = st.text_input('Enter a Prompt here')
process = st.button(label="Process")



if 'res' not in st.session_state:
    st.session_state.res = ""

if 'chain' not in st.session_state:
    st.session_state.chain = load_qa_chain(llm=llm, chain_type="stuff")


# if process:
if process:                                          

    # conversation =  load_qa_chain(llm=llm, chain_type="stuff")
    docs = VectorStore.similarity_search(query=prompt, k=5)
    llm_response =  st.session_state.chain.run(
                                                                question = prompt,
                                                                input_documents = docs
                                                         )   
    st.write(llm_response)
    st.session_state.res += llm_response
    #st.write(st.session_state.res)
    
if st.session_state.res is not []:
    st.write(st.session_state.res)

def clear_chat_history():
    st.session_state.res = []
# def initialize_session_state():

#     if "history" not in st.session_state:
#         st.session_state.history = []
 
#     if "conversation" not in st.session_state:
#         st.session_state.conversation =  load_qa_chain(llm=llm, chain_type="stuff")

#     if "human_prompt" not in st.session_state:
#         st.session_state.human_prompt = ""


# def on_click_callback():
   
#         human_prompt = st.session_state.human_prompt
#         st.session_state.human_prompt = ""
#       # for Doc answering 
#         if file_1:                                                                              
#             docs = VectorStore.similarity_search(query=human_prompt, k=5)
                                                                                     
#             llm_response = st.session_state.conversation.run(
#                                                                 question = human_prompt,
#                                                                     input_documents = docs
#                                                                 )   

#         st.session_state.history.append( Message("human", human_prompt) )
#         st.session_state.history.append(Message("ai", llm_response))

# initialize_session_state()


# with chat_placeholder:
#     print("Start")
#     print(st.session_state.history)
#     print("😀 😃 😄 😁 😆 😅 😂 🤣 🥲 🥹 ")
#     for chat in st.session_state.history:
#         print(chat)
#         div = f"""
# <div class="chat-row 
#     {'' if chat.origin == 'ai' else 'row-reverse'}">
#     <img class="chat-icon" src="app/static/{
#         'a_03.png' if chat.origin == 'ai' 
#                       else 'user_icon.png'}"
#          width=40 height=40>
#     <div class="chat-bubble
#     {'ai-bubble' if chat.origin == 'ai' else 'human-bubble'}">
#         &#8203;{chat.message}
#     </div>
# </div>
#         """
#         st.markdown(div, unsafe_allow_html=True)
    
#     for _ in range(3):
#         st.markdown("")


# with prompt_placeholder:
#     st.markdown("**Chat**")
#     cols = st.columns((4, 1))
#     cols[0].text_input(
#         "Chat",
#       #  value="Hello bot",
#         placeholder="Type your message here...",
#         label_visibility="collapsed",
#         key="human_prompt",
#     )
#     submit_clicked = cols[1].form_submit_button(
#         "Submit", 
#         type="primary", 
#         on_click=on_click_callback, 
#     )


End = False

if file_1:
    col1, col2 = st.columns(2)
    col1.text_area("Response")
    col2.text_area("Editable Response")

    rating = st.radio(
    "Rate the response",
    ["1", "2", "3", "4"],
    captions = ["Horrible","Poor", "Good", "Very Good"],
    horizontal=True,
    )
    save = st.button("save")
    if save:
        End = True

if End:
    filename = "testing_feed.csv"
    fields = ['Prompt', "LLM's Response",'Feedback', 'Rating', 'Document'] 
    rows = [ [prompt, st.session_state.res,'Feedback', rating, 'Doc link'], 
            ] 
    clear_chat_history()
    with open(filename, 'a') as csvfile: 
        csvwriter = csv.writer(csvfile)        
        #csvwriter.writerow(fields)         
        csvwriter.writerows(rows)
        


# if st.button("Clear Chat History"):
#     clear_chat_history()