# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import argparse
import os
import sys
from pathlib import Path
import torch
from loguru import logger
import torch.nn.functional as F
import ttnn
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from models.experimental.SSD512.reference.data.voc0712 import VOC_CLASSES
from models.experimental.SSD512.common import (
    build_and_init_torch_model,
    build_and_load_ttnn_model,
    generate_prior_boxes,
    synchronize_device,
    cleanup_device_memory,
)
from models.experimental.SSD512.tt.layers.detect import TtDetect


def load_image(image_path, size=512):
    """Load and preprocess image for SSD512 input."""
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not load image from {image_path}")

    original_height, original_width = image_bgr.shape[:2]

    original_img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_img = Image.fromarray(original_img_rgb)

    x = cv2.resize(image_bgr, (size, size)).astype(np.float32)

    resized_height, resized_width = x.shape[:2]
    if resized_height != size or resized_width != size:
        raise ValueError(f"Image resize failed: expected {size}x{size}, got {resized_height}x{resized_width}")

    x -= (104.0, 117.0, 123.0)

    x = x.astype(np.float32)

    img_tensor = torch.from_numpy(x).permute(2, 0, 1)

    img_tensor = img_tensor.unsqueeze(0)

    expected_shape = (1, 3, size, size)
    if img_tensor.shape != expected_shape:
        raise ValueError(f"Tensor shape mismatch: expected {expected_shape}, got {img_tensor.shape}")

    original_img.original_size = (original_width, original_height)

    return img_tensor, original_img


def filter_top_detections(detections, max_detections=2, min_score=0.1):
    """Filter detections to keep only top N by confidence score."""
    if len(detections) == 0:
        return detections

    # Get first image detections
    det = detections[0]
    boxes = det["boxes"]
    scores = det["scores"]
    labels = det["labels"]

    if len(boxes) == 0:
        return detections

    # Filter by minimum score first
    score_mask = scores >= min_score
    if not score_mask.any():
        # If no detections meet minimum score, return top max_detections anyway
        score_mask = torch.ones_like(scores, dtype=torch.bool)

    filtered_boxes = boxes[score_mask]
    filtered_scores = scores[score_mask]
    filtered_labels = labels[score_mask]

    if len(filtered_boxes) == 0:
        return detections

    # Sort by confidence score (descending)
    sorted_indices = torch.argsort(filtered_scores, descending=True)

    # Keep only top max_detections
    keep_indices = sorted_indices[:max_detections]

    filtered_detections = [
        {
            "boxes": filtered_boxes[keep_indices],
            "scores": filtered_scores[keep_indices],
            "labels": filtered_labels[keep_indices],
        }
    ]

    return filtered_detections


def draw_detections(image, detections, output_path, model_name):
    """Draw bounding boxes and labels on image."""
    draw = ImageDraw.Draw(image)

    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()

    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]

    if len(detections) > 0:
        det = detections[0]  # Get first image detections
        boxes = det["boxes"]
        scores = det["scores"]
        labels = det["labels"]

        if hasattr(image, "original_size"):
            img_width, img_height = image.original_size
        else:
            img_width, img_height = image.size
        scale = torch.tensor([img_width, img_height, img_width, img_height], dtype=torch.float32)

        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            box_scaled = (box * scale).cpu()
            x1 = box_scaled[0].item()
            y1 = box_scaled[1].item()
            x2 = box_scaled[2].item()
            y2 = box_scaled[3].item()

            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            x2 = max(0, min(x2, img_width))
            y2 = max(0, min(y2, img_height))

            if x1 >= x2 or y1 >= y2:
                continue

            class_idx = label.item()
            if class_idx < len(VOC_CLASSES):
                class_name = VOC_CLASSES[class_idx]
            else:
                class_name = f"Class {class_idx}"

            color = colors[class_idx % len(colors)]

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            # Draw label with background
            label_text = f"{class_name}: {score.item():.2f}"
            bbox = draw.textbbox((x1, y1 - 20), label_text, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, y1 - 20), label_text, fill=(255, 255, 255), font=font)

    # Add title
    title = f"{model_name} Detections"
    bbox = draw.textbbox((10, 10), title, font=font)
    draw.rectangle(bbox, fill=(0, 0, 0, 180))
    draw.text((10, 10), title, fill=(255, 255, 255), font=font)

    # Save image
    image.save(output_path)


def run_ttnn_detection(model, image_tensor, priors, device, conf_thresh=0.01, nms_thresh=0.45, top_k=200):
    """Run TTNN model and get detections."""
    # Verify image tensor shape before sending to backbone
    expected_shape = (1, 3, 512, 512)
    if image_tensor.shape != expected_shape:
        raise ValueError(f"Image tensor shape mismatch: expected {expected_shape}, got {image_tensor.shape}")

    # Synchronize and clear memory before forward pass
    synchronize_device(device)

    # Forward pass
    loc, conf = model.forward(image_tensor, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Convert TTNN tensors to torch tensors
    loc_torch = ttnn.to_torch(loc).float()
    conf_torch = ttnn.to_torch(conf).float()

    # Flatten tensors and reshape to [batch, num_priors, 4] and [batch, num_priors, num_classes]
    batch_size = 1
    loc_torch = loc_torch.reshape(batch_size, -1)  # Flatten to [1, total_elements]
    conf_torch = conf_torch.reshape(batch_size, -1)  # Flatten to [1, total_elements]

    num_priors = loc_torch.shape[1] // 4
    loc_torch = loc_torch.view(batch_size, num_priors, 4)
    conf_torch = conf_torch.view(batch_size, num_priors, model.num_classes)

    # Apply softmax to confidence
    conf_torch = F.softmax(conf_torch, dim=-1)

    # Convert priors to TTNN tensor
    priors_ttnn = ttnn.from_torch(priors, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Convert loc and conf to TTNN tensors
    loc_ttnn = ttnn.from_torch(loc_torch, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    conf_ttnn = ttnn.from_torch(conf_torch, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Create detect layer
    detect = TtDetect(
        num_classes=model.num_classes, top_k=top_k, conf_thresh=conf_thresh, nms_thresh=nms_thresh, device=device
    )

    # Get detections
    detections = detect(loc_ttnn, conf_ttnn, priors_ttnn)

    # Cleanup TTNN tensors
    if device is not None:
        ttnn.deallocate(loc_ttnn)
        ttnn.deallocate(conf_ttnn)
        ttnn.deallocate(priors_ttnn)
        synchronize_device(device)

    return detections


def main():
    parser = argparse.ArgumentParser(description="SSD512 Demo with Real Images")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/experimental/SSD512/resources/sample_output",
        help="Directory to save output images (default: ./models/experimental/SSD512/resources/sample_output)",
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.1,
        help="Confidence threshold for detections (default: 0.1, lower for random weights)",
    )
    parser.add_argument("--nms_thresh", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--top_k", type=int, default=5, help="Top K detections per class (default: 5)")
    parser.add_argument(
        "--max_detections", type=int, default=2, help="Maximum total detections across all classes (default: 2)"
    )
    parser.add_argument("--device_id", type=int, default=0, help="TTNN device ID (default: 0)")
    parser.add_argument(
        "--restart_device",
        action="store_true",
        help="Restart device between images to free all memory (slower but avoids OOM)",
    )
    parser.add_argument(
        "--l1_small_size",
        type=int,
        default=98304,
        help="L1 small size in bytes (default: 98304). Use 0 to use device default",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Build PyTorch model (only need to do this once for weight loading)
    torch_model = build_and_init_torch_model(phase="test", size=512, num_classes=21)

    # Initialize device and model (will be reused unless --restart_device is set)
    device = None
    ttnn_model = None

    def init_device_and_model():
        """Initialize device and model. Can be called multiple times if restarting device."""
        nonlocal device, ttnn_model

        if device is not None:
            ttnn.close_device(device)
            import gc

            gc.collect()

        # Initialize TTNN device
        if args.l1_small_size == 0:
            device = ttnn.open_device(device_id=args.device_id)
        else:
            l1_size = args.l1_small_size
            device = ttnn.open_device(device_id=args.device_id, l1_small_size=l1_size)

        # Build TTNN model and load weights
        ttnn_model = build_and_load_ttnn_model(torch_model, device, num_classes=21, weight_device=None)

        return device, ttnn_model

    # Initialize device and model once
    device, ttnn_model = init_device_and_model()

    try:
        # Generate prior boxes
        priors_torch = generate_prior_boxes()

        # Get image files
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(args.input_dir).glob(f"*{ext}"))
            image_files.extend(Path(args.input_dir).glob(f"*{ext.upper()}"))

        if len(image_files) == 0:
            raise ValueError(f"No images found in {args.input_dir}")

        # Process each image
        for img_idx, img_path in enumerate(image_files):
            # Restart device if requested (frees all memory including L1)
            if args.restart_device and img_idx > 0:
                device, ttnn_model = init_device_and_model()

            # Load and preprocess image
            image_tensor, original_img = load_image(str(img_path))

            # Run TTNN detection
            cleanup_device_memory(device)

            # Run TTNN detection
            try:
                ttnn_detections = run_ttnn_detection(
                    ttnn_model,
                    image_tensor,
                    priors_torch,
                    device,
                    conf_thresh=args.conf_thresh,
                    nms_thresh=args.nms_thresh,
                    top_k=args.top_k,
                )
            except RuntimeError as e:
                if "Out of Memory" in str(e) or "L1" in str(e):
                    continue
                else:
                    raise

            # Synchronize after TTNN detection to free memory
            synchronize_device(device)

            # Filter to top detections
            ttnn_detections = filter_top_detections(ttnn_detections, max_detections=args.max_detections, min_score=0.1)

            # Save TTNN output
            base_name = img_path.stem
            ttnn_output_path = os.path.join(args.output_dir, f"{base_name}_ttnn.jpg")
            draw_detections(original_img.copy(), ttnn_detections, ttnn_output_path, "SSD512")
            logger.info(f"Demo completed! Results saved to: {ttnn_output_path}")

    finally:
        # Close device
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
