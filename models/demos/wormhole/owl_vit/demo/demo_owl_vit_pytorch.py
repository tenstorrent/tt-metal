# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
OWL-ViT PyTorch Demo: Reference implementation for comparison.

This demo runs the original Hugging Face OWL-ViT model on CPU for verification
and visual comparison with the TTNN implementation.

Usage:
    python models/demos/wormhole/owl_vit/demo/demo_owl_vit_pytorch.py
"""

import sys
import time
from pathlib import Path

import requests
import torch
from loguru import logger
from PIL import Image, ImageDraw, ImageFont
from transformers import OwlViTForObjectDetection, OwlViTProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

# Constants
MODEL_NAME = "google/owlvit-base-patch32"
OUTPUT_DIR = Path(__file__).parent / "outputs"

# Use same colors as TTNN demo for easy comparison
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
    threshold: float = 0.1,
) -> Image.Image:
    """Draw bounding boxes on image with labels and scores."""
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


def run_pytorch_demo(
    image_source: str = "http://images.cocodataset.org/val2017/000000039769.jpg",
    text_queries: list[str] = None,
    output_name: str = "detection_result_pytorch.png",
    threshold: float = 0.1,
):
    """Run OWL-ViT inference using PyTorch."""
    if text_queries is None:
        text_queries = ["a cat", "a remote control", "a cushion"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("OWL-ViT PyTorch Reference Demo")
    logger.info("=" * 60)

    # Load model and processor
    logger.info(f"Loading model: {MODEL_NAME}")
    processor = OwlViTProcessor.from_pretrained(MODEL_NAME)
    model = OwlViTForObjectDetection.from_pretrained(MODEL_NAME)
    model.eval()

    # Load image
    logger.info(f"Loading image: {image_source}")
    image = load_image(image_source)

    # Prepare inputs
    inputs = processor(text=[text_queries], images=image, return_tensors="pt")

    # Run inference
    logger.info("Running inference...")
    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    inference_time = time.perf_counter() - start_time

    # Post-process
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs=outputs, threshold=threshold, target_sizes=target_sizes)[
        0
    ]

    boxes = results["boxes"].tolist()
    scores = results["scores"].tolist()
    labels = results["labels"].tolist()

    logger.info("=" * 60)
    logger.info("DETECTION RESULTS (PyTorch)")
    logger.info("=" * 60)
    logger.info(f"Inference time: {inference_time * 1000:.1f} ms")
    logger.info(f"Detected {len(boxes)} objects:")

    for box, score, label in zip(boxes, scores, labels):
        box_str = [round(x, 1) for x in box]
        logger.info(f"  {text_queries[label]}: score={score:.2f}, box={box_str}")

    # Draw results
    result_image = draw_boxes(image, boxes, scores, labels, text_queries, threshold=threshold)

    # Save output
    output_path = OUTPUT_DIR / output_name
    result_image.save(output_path)
    logger.info(f"Saved result to: {output_path}")

    logger.info("=" * 60)
    logger.info("DEMO COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_pytorch_demo()
