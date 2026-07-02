# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
YOLOX demo script for object detection on Tenstorrent hardware.
"""

import argparse
import torch
import ttnn
from PIL import Image
import numpy as np



def preprocess_image(image_path: str, input_size: int = 640) -> torch.Tensor:
    """
    Preprocess image for YOLOX inference.
    
    Args:
        image_path: Path to input image
        input_size: Model input size (default 640)
        
    Returns:
        Preprocessed tensor [1, 3, H, W]
    """
    image = Image.open(image_path).convert("RGB")
    
    # Resize with letterbox padding to maintain aspect ratio
    orig_w, orig_h = image.size
    scale = min(input_size / orig_w, input_size / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    
    image = image.resize((new_w, new_h), Image.BILINEAR)
    
    # Create padded image
    padded = Image.new("RGB", (input_size, input_size), (114, 114, 114))
    padded.paste(image, ((input_size - new_w) // 2, (input_size - new_h) // 2))
    
    # Convert to tensor and normalize
    img_array = np.array(padded).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor


def postprocess_predictions(outputs, conf_threshold: float = 0.25):
    """
    Postprocess YOLOX predictions to extract bounding boxes.
    
    Args:
        outputs: Model outputs
        conf_threshold: Confidence threshold for filtering
        
    Returns:
        List of detections [x1, y1, x2, y2, conf, class_id]
    """
    # Placeholder for actual postprocessing
    # In full implementation, this would decode predictions,
    # apply NMS, and convert to bounding boxes
    detections = []
    return detections


def main():
    parser = argparse.ArgumentParser(description="YOLOX Demo on Tenstorrent Hardware")
    parser.add_argument(
        "--image",
        type=str,
        default="sample.jpg",
        help="Path to input image"
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        default="yolox-s",
        choices=["yolox-nano", "yolox-tiny", "yolox-s", "yolox-m", "yolox-l", "yolox-x"],
        help="YOLOX model variant"
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for detections"
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="Tenstorrent device ID"
    )
    args = parser.parse_args()
    
    print(f"YOLOX Demo - Variant: {args.model_variant}")
    print(f"Input image: {args.image}")
    
    # Model configuration based on variant
    variant_configs = {
        "yolox-nano": {"depth": 0.33, "width": 0.25},
        "yolox-tiny": {"depth": 0.33, "width": 0.375},
        "yolox-s": {"depth": 0.33, "width": 0.50},
        "yolox-m": {"depth": 0.67, "width": 0.75},
        "yolox-l": {"depth": 1.0, "width": 1.0},
        "yolox-x": {"depth": 1.33, "width": 1.25},
    }
    config = variant_configs[args.model_variant]
    
    # Open device with try-finally for proper cleanup
    device = None
    try:
        device = ttnn.open_device(device_id=args.device_id)
        print(f"Opened device: {device}")
        
        # Preprocess image
        input_tensor = preprocess_image(args.image)
        print(f"Input shape: {input_tensor.shape}")
        
        # Convert to TTNN tensor
        tt_input = ttnn.from_torch(
            input_tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device
        )
        
        # Load model (placeholder - would load pretrained weights)
        from models.demos.yolox.tt.model_def import TtYOLOX
        
        model = TtYOLOX(
            device=device,
            num_classes=80,
            depth_multiplier=config["depth"],
            width_multiplier=config["width"],
            parameters=None  # Would load preprocessed weights
        )
        
        # Run inference
        print("Running inference...")
        outputs = model(tt_input)
        
        # Postprocess
        detections = postprocess_predictions(outputs, args.conf_threshold)
        print(f"Found {len(detections)} detections")
        
        for det in detections:
            print(f"  Box: {det[:4]}, Conf: {det[4]:.2f}, Class: {int(det[5])}")
        
        print("Demo completed successfully!")
        
    finally:
        if device is not None:
            ttnn.close_device(device)
            print("Device closed")


if __name__ == "__main__":
    main()
