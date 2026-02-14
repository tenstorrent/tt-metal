# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
except ImportError:
    print("Error: Could not import AutoImageProcessor or AutoModelForDepthEstimation from transformers.")
    print("Please ensure you have a recent version of transformers installed.")
    exit(1)

import os

from PIL import Image

import ttnn
from models.demos.depth_anything_v2.tt.model_def import TtDepthAnythingV2, custom_preprocessor


def run_demo(model_id="depth-anything/Depth-Anything-V2-Large-hf", image_path=None):
    print(f"Loading model: {model_id}")

    # 1. Load PyTorch Model (Reference)
    try:
        torch_model = AutoModelForDepthEstimation.from_pretrained(
            model_id, torch_dtype=torch.float32, trust_remote_code=True
        )
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
    print("converting weights...")
    parameters = custom_preprocessor(torch_model, "depth_anything_v2")
    tt_model = TtDepthAnythingV2(torch_model.config, parameters, device)
    print("TT Model initialized.")

    # 4. Prepare Input
    if image_path and os.path.exists(image_path):
        image = Image.open(image_path).convert("RGB")
    else:
        print("No image path provided or file missing, using a dummy input.")
        # Create a dummy image or use random tensor
        image = Image.new("RGB", (518, 518), color=(73, 109, 137))

    # Save input for reference
    image.save("input_image.png")

    inputs = image_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]  # (1, 3, 518, 518)

    # Convert pixel_values to ttnn
    # Note: Depth Anything expects 518x518 generally.
    # We use ROW_MAJOR_LAYOUT initially to avoid padding issues before patchification.
    tt_pixel_values = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # 6. Run Inference
    print("Running inference on Tenstorrent device...")
    try:
        if device is not None:
            predicted_depth = tt_model(tt_pixel_values)
            print("Inference completed successfully!")

            # Post-processing to save image
            # 1. To Torch
            predicted_depth = ttnn.to_torch(predicted_depth)

            # 2. Interpolate to original size
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            # 3. Normalize to 0-255
            output = prediction.detach().cpu().numpy()
            formatted = (output - output.min()) / (output.max() - output.min()) * 255.0
            formatted = formatted.astype("uint8")

            # 4. Save
            depth_image = Image.fromarray(formatted)
            output_path = "depth_map_output.png"
            depth_image.save(output_path)
            print(f"Depth map saved to {output_path}")

        else:
            print("Skipping inference because device is not available.")
    except Exception as e:
        print(f"Inference failed: {e}")

    # 6. Cleanup
    if device is not None:
        ttnn.close_device(device)
        print("Device closed.")


if __name__ == "__main__":
    run_demo()
