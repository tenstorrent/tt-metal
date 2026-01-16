# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import transformers
try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
except ImportError:
    print("Error: Could not import AutoImageProcessor or AutoModelForDepthEstimation from transformers.")
    print("Please ensure you have a recent version of transformers installed.")
    exit(1)

import ttnn
from models.demos.depth_anything_v2.tt.model_def import TtDepthAnythingV2, custom_preprocessor

def run_demo(
    model_id="depth-anything/Depth-Anything-V2-Large-hf",
    image_path="models/demos/depth_anything_v2/demo/demo_image.jpg"
):
    print(f"Loading model: {model_id}")
    
    # 1. Load PyTorch Model (Reference)
    try:
        torch_model = AutoModelForDepthEstimation.from_pretrained(model_id, torch_dtype=torch.float32, trust_remote_code=True)
        torch_model.eval()
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
    print("converting weights...")
    parameters = custom_preprocessor(torch_model, "depth_anything_v2")
    
    # Use the huggingface config directly as it has 'hidden_size', 'num_attention_heads' etc.
    tt_model = TtDepthAnythingV2(torch_model.config, parameters)
    print("TT Model initialized.")

    # 4. Cleanup
    if device is not None:
        ttnn.close_device(device)
        print("Device closed.")

if __name__ == "__main__":
    run_demo()
