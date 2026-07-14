# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
End-to-End Demo for SAM 2 (facebook/sam2-hiera-tiny) Image Mode on TTNN.
Opens Tenstorrent device, runs segmentation pipeline, validates output.
Follows owl_vit demo pattern — requires N150/N300 hardware.

Usage:
    python models/demos/sam2/demo/demo_segmentation.py
"""

import sys
from pathlib import Path

import torch
import ttnn
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models.demos.sam2.tt.sam2_model import TtnnSam2ImageModel


def main():
    """Open device, run SAM2 segmentation, verify output shape."""
    # Open device — will fail gracefully if no hardware
    try:
        device_id = 0
        device = ttnn.open_device(device_id=device_id)
        logger.info(f"Opened Tenstorrent device {device_id}")
    except Exception as e:
        logger.error(f"No Tenstorrent device available: {e}")
        logger.error("This demo requires N150/N300 hardware and a built tt-metalium.")
        sys.exit(1)

    try:
        logger.info("Initializing SAM2 (sam2-hiera-tiny) on device...")
        model = TtnnSam2ImageModel(device=device)

        # 1024x1024 image + 2 point prompts
        dummy_img = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)
        dummy_pts = torch.randn(1, 2, 2, dtype=torch.float32)

        logger.info(f"Input image shape: {dummy_img.shape}")
        logger.info(f"Input points shape: {dummy_pts.shape}")

        # Forward pass on device
        out = model.forward(image=dummy_img, points=dummy_pts)
        mask = out["pred_mask"]

        logger.info(f"Output mask shape: {mask.shape}")
        assert mask.shape == (1, 1, 256, 256), f"Unexpected mask shape: {mask.shape}"
        logger.info("✅ SAM2 single-image segmentation demo completed cleanly!")
        logger.info(f"IOU scores: {out['iou_scores']}")

    finally:
        ttnn.close_device(device)
        logger.info("Device closed.")


if __name__ == "__main__":
    main()
