# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""YOLOv3 demo script for object detection on Tenstorrent hardware."""

import argparse
import torch
import ttnn
from PIL import Image
import numpy as np


def preprocess_image(image_path: str, input_size: int = 416) -> torch.Tensor:
    """Preprocess image for YOLOv3 inference."""
    image = Image.open(image_path).convert("RGB")
    
    # Letterbox resize
    orig_w, orig_h = image.size
    scale = min(input_size / orig_w, input_size / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    
    image = image.resize((new_w, new_h), Image.BILINEAR)
    
    # Pad to square
    padded = Image.new("RGB", (input_size, input_size), (128, 128, 128))
    padded.paste(image, ((input_size - new_w) // 2, (input_size - new_h) // 2))
    
    # Normalize
    img_array = np.array(padded).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor


def main():
    parser = argparse.ArgumentParser(description="YOLOv3 Demo")
    parser.add_argument("--image", type=str, default="sample.jpg")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--conf_threshold", type=float, default=0.5)
    args = parser.parse_args()
    
    print(f"YOLOv3 Demo - Input: {args.image}")
    
    device = None
    try:
        device = ttnn.open_device(device_id=args.device_id)
        print(f"Opened device: {device}")
        
        # Preprocess
        input_tensor = preprocess_image(args.image)
        print(f"Input shape: {input_tensor.shape}")
        
        # Convert to TTNN
        tt_input = ttnn.from_torch(
            input_tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device
        )
        
        # Load model
        from models.demos.yolov3.tt.model_def import TtYoloV3
        
        model = TtYoloV3(device=device, num_classes=80, parameters={})
        
        # Inference
        print("Running inference...")
        outputs = model(tt_input)
        
        print(f"Got {len(outputs)} detection scales")
        print("Demo completed!")
        
    finally:
        if device is not None:
            ttnn.close_device(device)
            print("Device closed")


if __name__ == "__main__":
    main()
