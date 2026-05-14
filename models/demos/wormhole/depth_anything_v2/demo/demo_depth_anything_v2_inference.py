# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Depth-Anything-V2-Large Demo: Monocular Depth Estimation on Tenstorrent Hardware

This demo runs Depth-Anything-V2-Large for single-image depth estimation using
TTNN APIs on Wormhole N300 hardware.

Usage:
    python models/demos/wormhole/depth_anything_v2/demo/demo_depth_anything_v2_inference.py

Features:
    - Monocular depth estimation from a single RGB image
    - ViT-L/14 backbone running on TT hardware
    - Depth map visualization with color mapping
    - Side-by-side comparison with PyTorch reference
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from loguru import logger
from PIL import Image

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from models.demos.wormhole.depth_anything_v2.tt.depth_anything_v2_config import DepthAnythingV2Config
from models.demos.wormhole.depth_anything_v2.tt.ttnn_depth_anything_v2 import (
    preprocess_all_weights_for_ttnn,
    run_depth_anything_v2_inference,
)

# Constants
OUTPUT_DIR = Path(__file__).parent / "outputs"
IMAGE_SIZE = 518
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_and_preprocess_image(image_path: str, image_size: int = 518) -> tuple:
    """Load and preprocess image for Depth-Anything-V2.

    Args:
        image_path: Path or URL to input image
        image_size: Target image size

    Returns:
        preprocessed tensor [1, 3, H, W], original image (numpy), (orig_h, orig_w)
    """
    # Load image
    if image_path.startswith("http"):
        image = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")

    orig_h, orig_w = image.height, image.width

    # Resize to target size
    image_resized = image.resize((image_size, image_size), Image.BICUBIC)

    # Convert to numpy and normalize
    image_np = np.array(image_resized).astype(np.float32) / 255.0
    image_np = (image_np - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)

    # Convert to tensor [1, 3, H, W]
    pixel_values = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)

    return pixel_values, np.array(image), (orig_h, orig_w)


def visualize_depth(depth: np.ndarray, max_depth: float = None) -> np.ndarray:
    """Visualize depth map with color mapping.

    Args:
        depth: Depth map [H, W]
        max_depth: Optional max depth for normalization

    Returns:
        Color-mapped depth visualization [H, W, 3] (uint8)
    """
    if max_depth is None:
        max_depth = depth.max()

    if max_depth > 0:
        depth_normalized = (depth / max_depth * 255).astype(np.uint8)
    else:
        depth_normalized = np.zeros_like(depth, dtype=np.uint8)

    # Apply colormap (TURBO for better depth perception)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)
    return depth_colored


def run_demo(
    image_source: str = "http://images.cocodataset.org/val2017/000000039769.jpg",
    output_name: str = "depth_result.png",
    show_reference: bool = True,
):
    """
    Run the full Depth-Anything-V2 demo on TTNN.

    Args:
        image_source: URL or path to input image
        output_name: Name of output image file
        show_reference: Whether to include PyTorch reference comparison
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Depth-Anything-V2-Large Demo on Tenstorrent Hardware")
    logger.info("=" * 60)
    logger.info(f"Image: {image_source}")

    # Load and preprocess image
    logger.info("Loading and preprocessing image...")
    pixel_values, original_image, (orig_h, orig_w) = load_and_preprocess_image(image_source, IMAGE_SIZE)
    logger.info(f"Input shape: {pixel_values.shape}")
    logger.info(f"Original image size: {orig_w}x{orig_h}")

    # Load model
    logger.info("Loading PyTorch reference model...")
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
        from depth_anything_v2.dpt import DepthAnythingV2

        model = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
    except ImportError:
        try:
            from transformers import AutoModelForDepthEstimation

            model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return
    model.eval()

    config = DepthAnythingV2Config()

    # PyTorch reference (optional)
    ref_depth = None
    if show_reference:
        logger.info("Running PyTorch reference inference...")
        with torch.no_grad():
            ref_depth = model(pixel_values).cpu().numpy().squeeze()
        logger.info(f"Reference depth range: [{ref_depth.min():.4f}, {ref_depth.max():.4f}]")

    # TTNN inference
    logger.info("Opening TT device...")
    device = ttnn.open_device(device_id=0)

    try:
        logger.info("Preprocessing weights for TTNN...")
        parameters = preprocess_all_weights_for_ttnn(model, device, config)

        # Warm-up
        logger.info("Running warm-up inference...")
        _ = run_depth_anything_v2_inference(pixel_values, parameters, model, device, config)

        # Timed inference
        logger.info("Running timed inference...")
        start_time = time.perf_counter()
        ttnn_depth = run_depth_anything_v2_inference(pixel_values, parameters, model, device, config)
        inference_time = time.perf_counter() - start_time

        ttnn_depth_np = ttnn_depth.cpu().numpy().squeeze()

        logger.info("=" * 60)
        logger.info("INFERENCE RESULTS")
        logger.info("=" * 60)
        logger.info(f"Inference time: {inference_time * 1000:.1f} ms")
        logger.info(f"Throughput: {1.0 / inference_time:.2f} FPS")
        logger.info(f"TTNN depth range: [{ttnn_depth_np.min():.4f}, {ttnn_depth_np.max():.4f}]")
        logger.info(f"Depth map size: {ttnn_depth_np.shape}")

        if ref_depth is not None:
            from scipy.stats import pearsonr

            pcc, _ = pearsonr(ttnn_depth_np.flatten(), ref_depth.flatten())
            logger.info(f"PCC vs PyTorch: {pcc:.6f}")

        # Resize depth maps to original image size for visualization
        ttnn_depth_resized = cv2.resize(ttnn_depth_np, (orig_w, orig_h))

        # Visualize
        depth_colored = visualize_depth(ttnn_depth_resized)

        # Create comparison image
        if ref_depth is not None:
            ref_depth_resized = cv2.resize(ref_depth, (orig_w, orig_h))
            ref_colored = visualize_depth(ref_depth_resized)
            # Side by side: original | TTNN depth | PyTorch depth
            comparison = np.hstack([original_image, depth_colored, ref_colored])
        else:
            # Side by side: original | TTNN depth
            comparison = np.hstack([original_image, depth_colored])

        output_path = OUTPUT_DIR / output_name
        cv2.imwrite(str(output_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        logger.info(f"Saved result to: {output_path}")

    finally:
        ttnn.close_device(device)

    logger.info("=" * 60)
    logger.info("DEMO COMPLETE")
    logger.info("=" * 60)


# =============================================================================
# Pytest Entry Point
# =============================================================================


def test_depth_anything_v2_demo():
    """Run the Depth-Anything-V2 demo on TTNN."""
    run_demo()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Depth-Anything-V2 depth estimation demo on TTNN")
    parser.add_argument(
        "--image",
        type=str,
        default="http://images.cocodataset.org/val2017/000000039769.jpg",
        help="Path or URL to input image",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="depth_result.png",
        help="Output image filename",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Skip PyTorch reference comparison",
    )
    args = parser.parse_args()

    run_demo(
        image_source=args.image,
        output_name=args.output,
        show_reference=not args.no_reference,
    )
