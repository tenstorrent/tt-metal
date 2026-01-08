# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
OWL-ViT Demo: Zero-shot Object Detection on Tenstorrent Hardware

This demo shows how to run OWL-ViT for open-vocabulary object detection
using TTNN APIs on Wormhole (N150/N300) hardware.

Usage:
    pytest models/demos/wormhole/owl_vit/demo/demo_owl_vit_inference.py -v

Features:
    - Zero-shot text-conditioned object detection
    - Bounding box visualization
    - Performance benchmarking
"""

import sys
import time
from pathlib import Path

import pytest
import requests
import torch
from loguru import logger
from PIL import Image, ImageDraw, ImageFont
from transformers import OwlViTForObjectDetection, OwlViTProcessor

sys.path.insert(0, "/root/tt-metal")


# Constants
MODEL_NAME = "google/owlvit-base-patch32"
OUTPUT_DIR = Path("/root/tt-metal/models/demos/wormhole/owl_vit/demo/outputs")
COLORS = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 0, 0),  # Dark Red
    (0, 128, 0),  # Dark Green
]


def load_image(source: str) -> Image.Image:
    """Load image from URL or file path."""
    if source.startswith("http"):
        image = Image.open(requests.get(source, stream=True).raw)
    else:
        image = Image.open(source)
    return image.convert("RGB")


def draw_boxes(
    image: Image.Image,
    boxes: list,
    scores: list,
    labels: list,
    text_queries: list[str],
    threshold: float = 0.1,
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
    draw = ImageDraw.Draw(image)

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()

    for box, score, label in zip(boxes, scores, labels):
        if score < threshold:
            continue

        color = COLORS[label % len(COLORS)]
        x1, y1, x2, y2 = [int(x) for x in box]

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label
        label_text = f"{text_queries[label]}: {score:.2f}"
        text_bbox = draw.textbbox((x1, y1), label_text, font=font)
        text_height = text_bbox[3] - text_bbox[1]

        # Background for text
        draw.rectangle(
            [x1, y1 - text_height - 4, x1 + (text_bbox[2] - text_bbox[0]) + 4, y1],
            fill=color,
        )
        draw.text((x1 + 2, y1 - text_height - 2), label_text, fill="white", font=font)

    return image


def run_pytorch_inference(
    image: Image.Image,
    text_queries: list[str],
    threshold: float = 0.1,
):
    """
    Run OWL-ViT inference using PyTorch (reference implementation).

    Args:
        image: Input image
        text_queries: List of text queries
        threshold: Detection confidence threshold

    Returns:
        results: Detection results
        outputs: Raw model outputs
        inference_time: Time in seconds
    """
    processor = OwlViTProcessor.from_pretrained(MODEL_NAME)
    model = OwlViTForObjectDetection.from_pretrained(MODEL_NAME)
    model.eval()

    texts = [text_queries]
    inputs = processor(text=texts, images=image, return_tensors="pt")

    # Warm-up run
    with torch.no_grad():
        _ = model(**inputs)

    # Timed inference
    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    inference_time = time.perf_counter() - start_time

    # Post-process
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs=outputs,
        threshold=threshold,
        target_sizes=target_sizes,
    )

    return results[0], outputs, inference_time


def print_detection_results(results, text_queries: list[str]):
    """Print detection results in a formatted way."""
    boxes = results["boxes"]
    scores = results["scores"]
    labels = results["labels"]

    print(f"\n{'='*60}")
    print(f"DETECTION RESULTS ({len(boxes)} objects detected)")
    print(f"{'='*60}")

    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        box_coords = [round(x, 1) for x in box.tolist()]
        print(f"  {i+1}. {text_queries[label]}")
        print(f"      Confidence: {score.item():.3f}")
        print(f"      Box (x1,y1,x2,y2): {box_coords}")

    print(f"{'='*60}\n")


# ============================================================================
# Demo Tests
# ============================================================================


class TestOwlViTDemo:
    """Demo tests for OWL-ViT."""

    @pytest.fixture
    def setup_output_dir(self):
        """Create output directory for demo results."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return OUTPUT_DIR

    def test_basic_detection_demo(self, setup_output_dir):
        """
        Basic object detection demo.

        This test demonstrates:
        1. Loading an image
        2. Running OWL-ViT detection with text queries
        3. Visualizing results
        """
        logger.info("Starting OWL-ViT detection demo")

        # Load test image (cats on a couch)
        image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = load_image(image_url)
        logger.info(f"Loaded image: {image.size}")

        # Define text queries
        text_queries = [
            "a cat",
            "a remote control",
            "a couch",
        ]
        logger.info(f"Text queries: {text_queries}")

        # Run inference
        results, outputs, inference_time = run_pytorch_inference(image, text_queries, threshold=0.1)

        # Print results
        print_detection_results(results, text_queries)
        logger.info(f"Inference time: {inference_time*1000:.2f}ms")

        # Visualize
        image_with_boxes = image.copy()
        image_with_boxes = draw_boxes(
            image_with_boxes,
            results["boxes"].tolist(),
            results["scores"].tolist(),
            results["labels"].tolist(),
            text_queries,
        )

        # Save output
        output_path = setup_output_dir / "demo_basic_detection.png"
        image_with_boxes.save(output_path)
        logger.info(f"Saved visualization to {output_path}")

        # Assertions
        assert len(results["boxes"]) > 0, "Should detect at least one object"
        # Check that at least one cat was detected
        detected_labels = [text_queries[label] for label in results["labels"]]
        assert any("cat" in label for label in detected_labels), "Should detect a cat"

    def test_multi_query_detection(self, setup_output_dir):
        """
        Multi-query detection demo with various objects.
        """
        logger.info("Starting multi-query detection demo")

        # Load image
        image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = load_image(image_url)

        # Extended queries
        text_queries = [
            "a cat face",
            "cat ears",
            "a tv remote",
            "a blanket",
            "a cushion",
            "an animal",
        ]

        results, outputs, inference_time = run_pytorch_inference(image, text_queries, threshold=0.05)

        print_detection_results(results, text_queries)

        # Visualize
        image_with_boxes = image.copy()
        image_with_boxes = draw_boxes(
            image_with_boxes,
            results["boxes"].tolist(),
            results["scores"].tolist(),
            results["labels"].tolist(),
            text_queries,
            threshold=0.05,
        )

        output_path = setup_output_dir / "demo_multi_query_detection.png"
        image_with_boxes.save(output_path)
        logger.info(f"Saved visualization to {output_path}")

    def test_model_output_shapes(self):
        """
        Verify model output shapes for TTNN implementation reference.
        """
        logger.info("Checking model output shapes")

        processor = OwlViTProcessor.from_pretrained(MODEL_NAME)
        model = OwlViTForObjectDetection.from_pretrained(MODEL_NAME)

        image = load_image("http://images.cocodataset.org/val2017/000000039769.jpg")
        text_queries = ["a cat", "a dog"]

        texts = [text_queries]
        inputs = processor(text=texts, images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        print("\n" + "=" * 60)
        print("OUTPUT SHAPES FOR TTNN IMPLEMENTATION")
        print("=" * 60)
        print(f"Input pixel_values: {inputs['pixel_values'].shape}")
        print(f"  -> [batch, channels, height, width]")
        print(f"Input input_ids: {inputs['input_ids'].shape}")
        print(f"  -> [num_queries, seq_len]")
        print()
        print(f"Output logits: {outputs.logits.shape}")
        print(f"  -> [batch, num_patches, num_queries]")
        print(f"Output pred_boxes: {outputs.pred_boxes.shape}")
        print(f"  -> [batch, num_patches, 4] (cx, cy, w, h normalized)")

        if outputs.image_embeds is not None:
            print(f"Output image_embeds: {outputs.image_embeds.shape}")
        if outputs.text_embeds is not None:
            print(f"Output text_embeds: {outputs.text_embeds.shape}")

        print()
        print("Vision Model Internals:")
        model_config = model.config.vision_config
        print(f"  hidden_size: {model_config.hidden_size}")
        print(f"  num_attention_heads: {model_config.num_attention_heads}")
        print(f"  num_hidden_layers: {model_config.num_hidden_layers}")
        print(f"  intermediate_size: {model_config.intermediate_size}")
        print(f"  patch_size: {model_config.patch_size}")
        print(f"  image_size: {model_config.image_size}")
        num_patches = (model_config.image_size // model_config.patch_size) ** 2
        print(f"  num_patches: {num_patches}")

        print()
        print("Text Model Internals:")
        text_config = model.config.text_config
        print(f"  hidden_size: {text_config.hidden_size}")
        print(f"  num_attention_heads: {text_config.num_attention_heads}")
        print(f"  num_hidden_layers: {text_config.num_hidden_layers}")
        print(f"  intermediate_size: {text_config.intermediate_size}")
        print(f"  vocab_size: {text_config.vocab_size}")
        print(f"  max_position_embeddings: {text_config.max_position_embeddings}")

        print()
        print("Detection Heads:")
        print(f"  box_head.dense0: {model.box_head.dense0.weight.shape}")
        print(f"  box_head.dense1: {model.box_head.dense1.weight.shape}")
        print(f"  box_head.dense2: {model.box_head.dense2.weight.shape}")
        print(f"  class_head.dense0: {model.class_head.dense0.weight.shape}")
        # logit_scale and logit_shift are Linear layers in OWL-ViT
        if hasattr(model.class_head, "logit_scale") and hasattr(model.class_head.logit_scale, "weight"):
            print(f"  logit_scale: Linear{tuple(model.class_head.logit_scale.weight.shape)}")
        if hasattr(model.class_head, "logit_shift") and hasattr(model.class_head.logit_shift, "weight"):
            print(f"  logit_shift: Linear{tuple(model.class_head.logit_shift.weight.shape)}")
        print("=" * 60 + "\n")

        # Verify expected shapes
        assert outputs.logits.shape == torch.Size([1, 576, 2])
        assert outputs.pred_boxes.shape == torch.Size([1, 576, 4])

    def test_performance_benchmark(self):
        """
        Benchmark PyTorch inference performance (baseline for TTNN comparison).
        """
        logger.info("Running performance benchmark")

        processor = OwlViTProcessor.from_pretrained(MODEL_NAME)
        model = OwlViTForObjectDetection.from_pretrained(MODEL_NAME)
        model.eval()

        image = load_image("http://images.cocodataset.org/val2017/000000039769.jpg")
        text_queries = ["a cat", "a dog", "a remote control"]
        texts = [text_queries]
        inputs = processor(text=texts, images=image, return_tensors="pt")

        # Warm-up
        for _ in range(3):
            with torch.no_grad():
                _ = model(**inputs)

        # Benchmark
        num_iterations = 10
        times = []

        for _ in range(num_iterations):
            start = time.perf_counter()
            with torch.no_grad():
                outputs = model(**inputs)
            times.append(time.perf_counter() - start)

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print("\n" + "=" * 60)
        print("PYTORCH PERFORMANCE BENCHMARK (CPU)")
        print("=" * 60)
        print(f"Iterations: {num_iterations}")
        print(f"Average: {avg_time*1000:.2f}ms")
        print(f"Min: {min_time*1000:.2f}ms")
        print(f"Max: {max_time*1000:.2f}ms")
        print(f"Throughput: {1/avg_time:.2f} images/sec")
        print("=" * 60 + "\n")


def test_run_demo():
    """
    Entry point for running the full demo.

    Run with: pytest models/demos/wormhole/owl_vit/demo/demo_owl_vit_inference.py::test_run_demo -v -s
    """
    demo = TestOwlViTDemo()

    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run demos
    print("\n" + "=" * 80)
    print("OWL-ViT DEMO: Zero-shot Object Detection")
    print("=" * 80 + "\n")

    demo.test_basic_detection_demo(OUTPUT_DIR)
    demo.test_model_output_shapes()
    demo.test_performance_benchmark()

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print(f"Output images saved to: {OUTPUT_DIR}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    test_run_demo()
