# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
YOLO26 Object Detection Demo.

Demonstrates inference on Tenstorrent hardware with visualization.
"""

import argparse
import torch
import ttnn
import cv2
import numpy as np
from loguru import logger

# COCO class names
COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def load_image(image_path: str, input_size: int = 640) -> tuple:
    """
    Load and preprocess image for YOLO26.

    Args:
        image_path: Path to input image
        input_size: Model input size

    Returns:
        Tuple of (preprocessed_tensor, original_image, scale_info)
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    original_shape = image.shape[:2]  # (height, width)

    # Resize maintaining aspect ratio with padding
    scale = min(input_size / original_shape[0], input_size / original_shape[1])
    new_h, new_w = int(original_shape[0] * scale), int(original_shape[1] * scale)

    resized = cv2.resize(image, (new_w, new_h))

    # Pad to input_size x input_size
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    pad_h = (input_size - new_h) // 2
    pad_w = (input_size - new_w) // 2
    padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

    # Convert to tensor (NHWC, float32, normalized)
    tensor = torch.from_numpy(padded).float() / 255.0
    tensor = tensor.unsqueeze(0)  # Add batch dimension

    scale_info = {
        "original_shape": original_shape,
        "scale": scale,
        "pad_h": pad_h,
        "pad_w": pad_w,
    }

    return tensor, image, scale_info


def postprocess_detections(
    outputs: list,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    input_size: int = 640,
) -> list:
    """
    Post-process YOLO26 outputs to get detections.

    Args:
        outputs: List of output tensors from model
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS (if needed)
        input_size: Model input size

    Returns:
        List of detections: [(x1, y1, x2, y2, confidence, class_id), ...]
    """
    detections = []

    # Process each scale
    strides = [8, 16, 32]  # P3, P4, P5 strides

    for scale_idx, (output, stride) in enumerate(zip(outputs, strides)):
        if isinstance(output, ttnn.Tensor):
            output = ttnn.to_torch(output)

        batch, h, w, channels = output.shape
        num_classes = channels - 4  # channels = num_classes + 4 (bbox)

        # Reshape to [batch, h*w, channels]
        output = output.reshape(batch, h * w, channels)

        # Split into bbox and class predictions
        bbox = output[..., :4]  # [batch, h*w, 4]
        cls_scores = output[..., 4:]  # [batch, h*w, num_classes]

        # Get max class score and index
        cls_max, cls_idx = cls_scores.max(dim=-1)

        # Filter by confidence
        mask = cls_max > conf_threshold

        for b in range(batch):
            valid_indices = mask[b].nonzero(as_tuple=True)[0]

            for idx in valid_indices:
                idx = idx.item()
                score = cls_max[b, idx].item()
                class_id = cls_idx[b, idx].item()

                # Decode bbox (cx, cy, w, h) -> (x1, y1, x2, y2)
                cx, cy, bw, bh = bbox[b, idx].tolist()

                # Convert from grid coordinates to image coordinates
                grid_x = idx % w
                grid_y = idx // w
                cx = (grid_x + cx) * stride
                cy = (grid_y + cy) * stride
                bw = bw * stride
                bh = bh * stride

                x1 = cx - bw / 2
                y1 = cy - bh / 2
                x2 = cx + bw / 2
                y2 = cy + bh / 2

                detections.append([x1, y1, x2, y2, score, class_id])

    return detections


def draw_detections(image: np.ndarray, detections: list, scale_info: dict) -> np.ndarray:
    """
    Draw detection boxes on image.

    Args:
        image: Original image
        detections: List of detections
        scale_info: Scale and padding info

    Returns:
        Image with drawn detections
    """
    output_image = image.copy()
    scale = scale_info["scale"]
    pad_h = scale_info["pad_h"]
    pad_w = scale_info["pad_w"]

    for det in detections:
        x1, y1, x2, y2, score, class_id = det

        # Remove padding and scale back to original size
        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale
        y2 = (y2 - pad_h) / scale

        # Clip to image bounds
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(image.shape[1], int(x2))
        y2 = min(image.shape[0], int(y2))

        # Draw box
        color = (0, 255, 0)  # Green
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)

        # Draw label
        class_id = int(class_id)
        if class_id < len(COCO_CLASSES):
            label = f"{COCO_CLASSES[class_id]}: {score:.2f}"
        else:
            label = f"Class {class_id}: {score:.2f}"

        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(output_image, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
        cv2.putText(output_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return output_image


def main():
    parser = argparse.ArgumentParser(description="YOLO26 Object Detection Demo")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input image path")
    parser.add_argument("--output", "-o", type=str, default="output.jpg", help="Output image path")
    parser.add_argument("--variant", type=str, default="yolo26n", help="Model variant")
    parser.add_argument("--input-size", type=int, default=640, help="Input size")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device-id", type=int, default=0, help="Device ID")
    args = parser.parse_args()

    logger.info(f"Running YOLO26 demo: {args.variant}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Input size: {args.input_size}")

    # Load image
    input_tensor, original_image, scale_info = load_image(args.input, args.input_size)
    logger.info(f"Loaded image: {original_image.shape}")

    # Initialize device
    device = ttnn.open_device(device_id=args.device_id)
    logger.info(f"Opened device: {device}")

    try:
        # Load model
        from models.experimental.yolo26.tt.ttnn_yolo26 import TtYOLO26

        logger.info("Loading YOLO26 model...")
        tt_model = TtYOLO26(device, args.variant)

        # Try to load weights from ultralytics
        try:
            tt_model.load_weights_from_ultralytics(args.variant)
            logger.info("Loaded weights from Ultralytics")
        except Exception as e:
            logger.warning(f"Could not load from Ultralytics: {e}")
            logger.info("Please run setup.sh first to download weights")
            return

        # Convert input to TTNN
        tt_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        logger.info(f"Input tensor shape: {input_tensor.shape}")

        # Run inference
        logger.info("Running inference...")
        outputs = tt_model(tt_input)
        logger.info(f"Got {len(outputs)} output scales")

        # Post-process
        detections = postprocess_detections(outputs, conf_threshold=args.conf_threshold, input_size=args.input_size)
        logger.info(f"Found {len(detections)} detections")

        # Draw results
        output_image = draw_detections(original_image, detections, scale_info)

        # Save output
        cv2.imwrite(args.output, output_image)
        logger.info(f"Saved output to: {args.output}")

        # Print detections
        for det in detections:
            x1, y1, x2, y2, score, class_id = det
            class_id = int(class_id)
            class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"Class {class_id}"
            logger.info(f"  {class_name}: {score:.2f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

    finally:
        ttnn.close_device(device)
        logger.info("Closed device")


if __name__ == "__main__":
    main()
