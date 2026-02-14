# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
OWL-ViT Demo: Zero-shot Object Detection on Tenstorrent Hardware

This demo runs OWL-ViT for open-vocabulary object detection using TTNN APIs
on Wormhole N300 hardware.

Usage:
    python models/demos/wormhole/owl_vit/demo/demo_owl_vit_inference.py

Features:
    - Zero-shot text-conditioned object detection
    - Full pipeline running on TT hardware (vision encoder, text encoder, detection heads)
    - Bounding box visualization with labels
"""

import sys
import time
from pathlib import Path

import requests
import torch
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

# Import TTNN implementation from tests
from models.demos.wormhole.owl_vit.tests.test_end_to_end import (
    get_pytorch_model_and_inputs,
    preprocess_all_weights_for_ttnn,
    run_owl_vit_end_to_end,
)

# Constants
OUTPUT_DIR = Path(__file__).parent / "outputs"
DETECTION_THRESHOLD = 0.3

COLORS = [
    (255, 50, 50),  # Red
    (50, 200, 50),  # Green
    (50, 50, 255),  # Blue
    (255, 200, 50),  # Yellow
    (255, 50, 255),  # Magenta
    (50, 255, 255),  # Cyan
]


def load_image(source: str) -> Image.Image:
    """Load image from URL or file path."""
    if source.startswith("http"):
        return Image.open(requests.get(source, stream=True).raw).convert("RGB")
    return Image.open(source).convert("RGB")


def draw_boxes(
    image: Image.Image,
    boxes: list,
    scores: list,
    labels: list,
    text_queries: list[str],
    threshold: float = 0.3,
) -> Image.Image:
    """
    Draw bounding boxes on image with labels and scores.

    Args:
        image: PIL Image
        boxes: List of boxes in [x1, y1, x2, y2] format
        scores: Confidence scores
        labels: Label indices
        text_queries: Text query strings
        threshold: Score threshold for display

    Returns:
        Image with drawn boxes
    """
    image = image.copy()
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for box, score, label in zip(boxes, scores, labels):
        if score < threshold:
            continue

        x1, y1, x2, y2 = [int(x) for x in box]
        color = COLORS[label % len(COLORS)]
        label_text = f"{text_queries[label]}: {score:.2f}"

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

        # Draw label background
        text_bbox = draw.textbbox((x1, y1 - 22), label_text, font=font)
        draw.rectangle(
            [text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2],
            fill=color,
        )
        draw.text((x1, y1 - 22), label_text, fill="white", font=font)

    return image


def apply_nms(boxes, scores, labels, iou_threshold: float = 0.5, max_detections: int = 10):
    """Apply simple non-maximum suppression to reduce overlapping boxes."""

    def compute_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (area1 + area2 - inter + 1e-6)

    # Combine and sort by score
    detections = list(zip(boxes, scores, labels))
    detections.sort(key=lambda x: x[1], reverse=True)

    kept = []
    for box, score, label in detections:
        overlap = False
        for kept_box, _, _ in kept:
            if compute_iou(box, kept_box) > iou_threshold:
                overlap = True
                break
        if not overlap:
            kept.append((box, score, label))
        if len(kept) >= max_detections:
            break

    if not kept:
        return [], [], []

    kept_boxes, kept_scores, kept_labels = zip(*kept)
    return list(kept_boxes), list(kept_scores), list(kept_labels)


def run_ttnn_inference(
    image: Image.Image,
    text_queries: list[str],
    device: ttnn.Device,
    threshold: float = 0.3,
):
    """
    Run OWL-ViT inference using TTNN on TT hardware.

    Args:
        image: Input image
        text_queries: List of text queries for detection
        device: TTNN device
        threshold: Detection confidence threshold

    Returns:
        boxes: List of bounding boxes [x1, y1, x2, y2]
        scores: Confidence scores
        labels: Label indices
        inference_time: Time in seconds
    """
    # Load model and processor
    processor, model, inputs, _ = get_pytorch_model_and_inputs(text_queries, image)

    # Preprocess weights for TTNN
    logger.info("Loading model weights to device...")
    parameters = preprocess_all_weights_for_ttnn(model, device)

    # Warm-up run
    logger.info("Running warm-up inference...")
    _ = run_owl_vit_end_to_end(
        inputs["pixel_values"],
        inputs["input_ids"],
        inputs["attention_mask"],
        parameters,
        device,
        model,
    )

    # Timed inference
    logger.info("Running timed inference...")
    start_time = time.perf_counter()
    pred_boxes, logits = run_owl_vit_end_to_end(
        inputs["pixel_values"],
        inputs["input_ids"],
        inputs["attention_mask"],
        parameters,
        device,
        model,
    )
    inference_time = time.perf_counter() - start_time

    # Convert outputs to torch
    ttnn_boxes_torch = ttnn.to_torch(pred_boxes)
    ttnn_logits_torch = ttnn.to_torch(logits)

    # Create output object for HuggingFace post-processing
    class TTNNOutputs:
        def __init__(self, logits, pred_boxes):
            self.logits = logits
            self.pred_boxes = pred_boxes

    ttnn_output_obj = TTNNOutputs(ttnn_logits_torch, ttnn_boxes_torch)
    target_sizes = torch.Tensor([image.size[::-1]])

    # Post-process
    results = processor.post_process_object_detection(
        outputs=ttnn_output_obj,
        threshold=threshold,
        target_sizes=target_sizes,
    )

    boxes = results[0]["boxes"].tolist()
    scores = results[0]["scores"].tolist()
    labels = results[0]["labels"].tolist()

    # Apply NMS
    boxes, scores, labels = apply_nms(boxes, scores, labels)

    return boxes, scores, labels, inference_time


def run_demo(
    image_source: str = "http://images.cocodataset.org/val2017/000000039769.jpg",
    text_queries: list[str] = None,
    output_name: str = "detection_result.png",
    threshold: float = 0.1,
):
    """
    Run the full OWL-ViT demo on TTNN.

    Args:
        image_source: URL or path to input image
        text_queries: List of text queries for detection
        output_name: Name of output image file
    """
    if text_queries is None:
        text_queries = ["a cat", "a remote control", "a cushion"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("OWL-ViT Demo on Tenstorrent Hardware")
    logger.info("=" * 60)
    logger.info(f"Image: {image_source}")
    logger.info(f"Queries: {text_queries}")

    # Load image
    logger.info("Loading image...")
    image = load_image(image_source)
    logger.info(f"Image size: {image.size}")

    # Open device and run inference
    logger.info("Opening TT device...")
    device = ttnn.open_device(device_id=0)

    try:
        # Use lower threshold to find smaller/occluded objects
        boxes, scores, labels, inference_time = run_ttnn_inference(image, text_queries, device, threshold=threshold)

        logger.info("=" * 60)
        logger.info("DETECTION RESULTS (TTNN)")
        logger.info("=" * 60)
        logger.info(f"Inference time: {inference_time * 1000:.1f} ms")
        logger.info(f"Detected {len(boxes)} objects:")

        for box, score, label in zip(boxes, scores, labels):
            box_str = [round(x, 1) for x in box]
            logger.info(f"  {text_queries[label]}: score={score:.2f}, box={box_str}")

        # Draw results with the lower threshold
        result_image = draw_boxes(image, boxes, scores, labels, text_queries, threshold=threshold)

        # Save output
        output_path = OUTPUT_DIR / output_name
        result_image.save(output_path)
        logger.info(f"Saved result to: {output_path}")

    finally:
        ttnn.close_device(device)

    logger.info("=" * 60)
    logger.info("DEMO COMPLETE")
    logger.info("=" * 60)


# =============================================================================
# Pytest Entry Points
# =============================================================================


def test_owl_vit_demo():
    """
    Run the OWL-ViT demo on TTNN.

    This test demonstrates the full detection pipeline running on TT hardware.
    """
    run_demo()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run OWL-ViT object detection demo on TTNN")
    parser.add_argument(
        "--image",
        type=str,
        default="http://images.cocodataset.org/val2017/000000039769.jpg",
        help="Path or URL to input image",
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=["a cat", "a remote control", "a cushion"],
        help="List of text queries to detect",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="detection_result.png",
        help="Output image filename",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Detection confidence threshold",
    )
    args = parser.parse_args()

    run_demo(
        image_source=args.image,
        text_queries=args.queries,
        output_name=args.output,
        threshold=args.threshold,
    )
