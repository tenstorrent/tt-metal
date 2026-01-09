# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import transformers

try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
except ImportError:
    print("Error: Could not import AutoImageProcessor or AutoModelForDepthEstimation from transformers.")
    print("Please ensure you have a recent version of transformers installed.")
    exit(1)

import ttnn
from models.demos.depth_anything_v2.tt.model_def import TtDepthAnythingV2

def run_demo(
    model_id="depth-anything/Depth-Anything-V2-Large-hf",
    image_path="models/demos/depth_anything_v2/demo/demo_image.jpg"
):
    print(f"Loading model: {model_id}")
    
    # 1. Load PyTorch Model (Reference)
    try:
        torch_model = AutoModelForDepthEstimation.from_pretrained(model_id, torch_dtype=torch.float32, trust_remote_code=True)
        torch_model.eval()
        image_processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
        print("PyTorch model loaded successfully.")
    except Exception as e:
        print(f"Failed to load Hugging Face model: {e}")
        return

    # 2. Initialize Tenstorrent Device
    device = None
    try:
        device_id = 0
        device = ttnn.CreateDevice(device_id=device_id)
        print(f"Device {device_id} opened.")
    except Exception as e:
        print(f"Warning: Failed to open Tenstorrent device (HW not present?): {e}")
        print("Proceeding to verify model weight conversion on host...")

    # 3. Convert Weights & Initialize TT Model
    from models.demos.depth_anything_v2.tt.model_def import custom_preprocessor, TtDepthAnythingV2
    
    print("converting weights...")
    parameters = custom_preprocessor(torch_model, "depth_anything_v2")
    
    # Needs a config object as expected by model_def.py
    # We can use the huggingface config directly as it has 'hidden_size', 'num_attention_heads' etc.
    tt_model = TtDepthAnythingV2(parameters)
    print("TT Model initialized.")

    # 4. Load & Preprocess Image
    # ...
    
    # 5. Run Inference
    # input_tensor = ...
    # output = tt_model(input_tensor)
    
    # Limit cleanup
    ttnn.close_device(device)
    print("Device closed.")

if __name__ == "__main__":
    run_demo()
