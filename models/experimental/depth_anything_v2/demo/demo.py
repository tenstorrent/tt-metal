# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Demo script for Depth Anything V2 Large on Tenstorrent Wormhole.

Usage:
    python models/experimental/depth_anything_v2/demo/demo.py
    python models/experimental/depth_anything_v2/demo/demo.py --model_id "depth-anything/Depth-Anything-V2-Large-hf"
    python models/experimental/depth_anything_v2/demo/demo.py --image_path /path/to/image.jpg
"""

import argparse
import os
import traceback

import torch

try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
except ImportError:
    print("Error: Could not import AutoImageProcessor or AutoModelForDepthEstimation from transformers.")
    print("Please ensure you have a recent version of transformers installed.")
    exit(1)

from PIL import Image

import ttnn
from models.experimental.depth_anything_v2.tt.model_def import TtDepthAnythingV2, custom_preprocessor


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
    try:
        device_id = 0
        device = ttnn.open_device(device_id=device_id, l1_small_size=32768)
        print(f"Device {device_id} opened.")
    except Exception as e:
        print(f"ERROR: Failed to open Tenstorrent device: {e}")
        print("A Tenstorrent Wormhole device is required to run this demo.")
        return

    # 3. Convert Weights & Initialize TT Model
    print("Converting weights...")
    parameters = custom_preprocessor(torch_model, "depth_anything_v2")
    tt_model = TtDepthAnythingV2(torch_model.config, parameters, device)
    print("TT Model initialized.")

    # 4. Prepare Input
    if image_path and os.path.exists(image_path):
        image = Image.open(image_path).convert("RGB")
    else:
        if image_path:
            print(f"Warning: Image path '{image_path}' not found, using a dummy input.")
        else:
            print("No image path provided, using a dummy input.")
        image = Image.new("RGB", (518, 518), color=(73, 109, 137))

    # Save input for reference
    image.save("input_image.png")

    inputs = image_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]  # (1, 3, 518, 518)

    # Convert pixel_values to ttnn
    tt_pixel_values = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # 5. Run PyTorch reference inference
    print("Running PyTorch reference inference...")
    with torch.no_grad():
        torch_output = torch_model(pixel_values).predicted_depth

    ref_np = torch_output.squeeze().cpu().numpy()
    ref_formatted = (ref_np - ref_np.min()) / (ref_np.max() - ref_np.min() + 1e-8) * 255.0
    ref_formatted = ref_formatted.astype("uint8")
    ref_depth_image = Image.fromarray(ref_formatted)
    ref_depth_image.save("depth_map_output_reference.png")
    print("PyTorch reference depth map saved to depth_map_output_reference.png")

    # 6. Run TTNN inference on Tenstorrent device
    print("Running inference on Tenstorrent device...")
    try:
        predicted_depth = tt_model(tt_pixel_values)
        print("Inference completed successfully!")

        # Post-processing to save image
        predicted_depth = ttnn.to_torch(predicted_depth)

        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        tt_np = prediction.detach().cpu().numpy()
        tt_formatted = (tt_np - tt_np.min()) / (tt_np.max() - tt_np.min() + 1e-8) * 255.0
        tt_formatted = tt_formatted.astype("uint8")

        tt_depth_image = Image.fromarray(tt_formatted)
        tt_depth_image.save("depth_map_output_ttnn.png")
        print("TTNN depth map saved to depth_map_output_ttnn.png")

        # 7. Create side-by-side comparison: Input | PyTorch Reference | TTNN
        w, h = 256, 256
        input_resized = image.resize((w, h))
        ref_resized = ref_depth_image.resize((w, h)).convert("RGB")
        tt_resized = tt_depth_image.resize((w, h)).convert("RGB")

        comparison = Image.new("RGB", (w * 3, h))
        comparison.paste(input_resized, (0, 0))
        comparison.paste(ref_resized, (w, 0))
        comparison.paste(tt_resized, (w * 2, 0))
        comparison.save("depth_map_comparison.png")
        print("Side-by-side comparison saved to depth_map_comparison.png")

    except Exception as e:
        print(f"Inference failed: {e}")
        traceback.print_exc()

    # 8. Cleanup
    ttnn.close_device(device)
    print("Device closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth Anything V2 Large demo on Tenstorrent Wormhole")
    parser.add_argument(
        "--model_id",
        type=str,
        default="depth-anything/Depth-Anything-V2-Large-hf",
        help="HuggingFace model identifier (default: depth-anything/Depth-Anything-V2-Large-hf)",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to input image (default: uses a dummy 518x518 image)",
    )
    args = parser.parse_args()
    run_demo(model_id=args.model_id, image_path=args.image_path)
