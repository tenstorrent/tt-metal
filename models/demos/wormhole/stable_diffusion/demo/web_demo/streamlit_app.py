# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import io
import time

import requests
import streamlit as st
from PIL import Image

# parse args
parser = argparse.ArgumentParser(description="Streamlit app for Stable Diffusion")
parser.add_argument("--port", type=int, default=7000, help="The port number the Streamlit server should run on")
args = parser.parse_args()

# Add space below the title
st.set_page_config(page_title="Stable-Diffusion-1.4")
st.title("TT Stable Diffusion Playground")
st.markdown("<br>", unsafe_allow_html=True)

# Display an image logo at the end of the title

# Popover for Device ID input
default_device_id = f"localhost:{args.port}"  # default URL
with st.expander("Override API URL"):
    device_id = st.text_input("What's your API URL? e.x. 10.229.36.110:5000")
    device_id = device_id or default_device_id

st.markdown("<br>", unsafe_allow_html=True)

prompt = st.text_input("Enter your prompt:", "")


# Function to save image to Downloads and display it
def save_and_display_image(image_data):
    try:
        image = Image.open(io.BytesIO(image_data))
        image_placeholder.image(image, use_container_width=True)
    except Exception as e:
        st.error(f"Error processing image: {e}")


# Placeholder for the image
image_placeholder = st.empty()


# Function to check and update the image
def check_and_update_image(server_url, task_id):
    global curr
    try:
        status_response = requests.get(f"{server_url}/status/{task_id}")
        if status_response.json().get("status") == "Completed":
            image_response = requests.get(f"{server_url}/fetch_image/{task_id}")
            save_and_display_image(image_response.content)
            return True
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to the server: {e}")


# generate image if prompt was entered
if prompt:
    if not device_id:
        st.error("Please enter your device ID.")
    else:
        SERVER_URL = f"http://{device_id}"
        with st.spinner("Running Stable Diffusion"):
            try:
                data = {"prompt": prompt}
                response = requests.post(f"{SERVER_URL}/enqueue", json=data, timeout=2)
                if response.status_code != 201:
                    st.error(f"Error submitting prompt: {response.status_code} - {response.text}")
                task_id = response.json().get("task_id")
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to the server: {e}")
            else:
                while not (updated := check_and_update_image(SERVER_URL, task_id)):
                    time.sleep(2)

hide_decoration_bar_style = """
    <style>
        header {visibility: hidden;}
    </style>
"""
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)
