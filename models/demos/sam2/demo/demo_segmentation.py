# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
End-to-End Demo for SAM 2 (facebook/sam2-hiera-tiny) Image Mode on TTNN.
Opens Tenstorrent device, runs segmentation pipeline, validates output.
Matches HF Sam2Model architecture — requires N150/N300 hardware.

Usage:
    python models/demos/sam2/demo/demo_segmentation.py
"""

import sys
from pathlib import Path
import urllib.request, json

import torch
import ttnn
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, '/tmp')

from models.demos.sam2.tt.sam2_model import TtnnSam2Model


def load_sam2_config():
    """Load sam2-hiera-tiny config from HuggingFace."""
    url = 'https://huggingface.co/facebook/sam2-hiera-tiny/raw/main/config.json'
    return json.loads(urllib.request.urlopen(url).read().decode())


def main():
    """Open device, run SAM2 segmentation, verify output."""
    # Open device
    try:
        device_id = 0
        device = ttnn.open_device(device_id=device_id)
        logger.info(f"Opened Tenstorrent device {device_id}")
    except Exception as e:
        logger.error(f"No Tenstorrent device available: {e}")
        logger.error("This demo requires N150/N300 hardware and a built tt-metalium.")
        sys.exit(1)

    try:
        cfg = load_sam2_config()
        logger.info("Initializing SAM2 (sam2-hiera-tiny) on device...")

        model = TtnnSam2Model(
            device=device,
            vision_config=cfg.get("vision_config", {}),
            prompt_config=cfg.get("prompt_encoder_config", {}),
            mask_decoder_config=cfg.get("mask_decoder_config", {}),
        )

        # 1024x1024 image + single point prompt at center
        dummy_img = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)
        dummy_pts = torch.tensor([[[[512.0, 512.0]]]], dtype=torch.float32)
        dummy_labels = torch.ones(1, 1, 1, dtype=torch.int32)

        logger.info(f"Input image shape: {dummy_img.shape}")
        logger.info(f"Input points shape: {dummy_pts.shape}")

        # Forward pass on device
        out = model.forward(
            pixel_values=dummy_img,
            input_points=dummy_pts,
            input_labels=dummy_labels,
        )
        masks = out["pred_masks"]
        iou = out["iou_scores"]

        logger.info(f"Output mask shape: {masks.shape}")
        logger.info(f"Output IoU scores: {iou}")
        assert masks is not None, "Mask output is None"
        assert iou is not None, "IoU output is None"
        logger.info("✅ SAM2 single-image segmentation demo completed cleanly!")

    finally:
        ttnn.close_device(device)
        logger.info("Device closed.")


if __name__ == "__main__":
    main()
