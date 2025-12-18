# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
SmolVLA nightly CI test - runs PCC verification for 1-step inference.

This test validates that TT implementation matches CPU within expected precision tolerances.
"""

import numpy as np
import torch
from loguru import logger
from PIL import Image


def compute_pcc(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson Correlation Coefficient between two arrays."""
    x_flat = torch.as_tensor(x).flatten().float()
    y_flat = torch.as_tensor(y).flatten().float()
    x_centered = x_flat - x_flat.mean()
    y_centered = y_flat - y_flat.mean()
    numerator = (x_centered * y_centered).sum()
    denominator = torch.sqrt((x_centered**2).sum() * (y_centered**2).sum())
    return (numerator / denominator).item() if denominator != 0 else 0.0


def create_test_image(size: int = 512, seed: int = 42) -> Image.Image:
    """Create a deterministic test image."""
    np.random.seed(seed)
    img_array = np.zeros((size, size, 3), dtype=np.uint8)
    img_array[:, :, 0] = np.tile(np.linspace(50, 200, size), (size, 1)).astype(np.uint8)
    img_array[:, :, 1] = np.tile(np.linspace(100, 150, size), (size, 1)).T.astype(np.uint8)
    img_array[:, :, 2] = 128
    return Image.fromarray(img_array)


def test_smolvla_pcc(device):
    """
    Run SmolVLA PCC verification test.

    Tests 1-step inference PCC between CPU and TT implementations.
    PCC threshold: 0.90 (expected for bfloat8_b precision)
    """
    import ttnn
    from models.experimental.smolvla.tt.smol_vla import SmolVLAForActionPrediction

    logger.info("Running SmolVLA PCC test")

    try:
        # Load CPU model
        logger.info("Loading CPU model...")
        model_cpu = SmolVLAForActionPrediction.from_pretrained("lerobot/smolvla_base", ttnn_device=None)
        model_cpu.processor.image_processor.do_image_splitting = False
        model_cpu.eval()

        # Load TT model (use the device fixture)
        logger.info("Loading TT model...")
        model_tt = SmolVLAForActionPrediction.from_pretrained("lerobot/smolvla_base", ttnn_device=device)
        model_tt.processor.image_processor.do_image_splitting = False
        model_tt.eval()

        # Create test inputs
        img = create_test_image()
        instruction = "pick up the red block"

        # Run CPU inference
        logger.info("Running CPU inference...")
        torch.manual_seed(42)
        np.random.seed(42)
        actions_cpu = model_cpu.predict_action(
            images=[img], instruction=instruction, num_inference_steps=1, action_dim=6
        )

        # Run TT inference
        logger.info("Running TT inference...")
        torch.manual_seed(42)
        np.random.seed(42)
        actions_tt = model_tt.predict_action(images=[img], instruction=instruction, num_inference_steps=1, action_dim=6)

        # Compute PCC
        actions_cpu_np = np.asarray(actions_cpu)
        actions_tt_np = np.asarray(actions_tt)
        pcc = compute_pcc(actions_cpu_np, actions_tt_np)

        logger.info(f"1-Step Inference PCC: {pcc:.4f}")
        logger.info(
            f"CPU output: [{actions_cpu_np[0,0]:.4f}, {actions_cpu_np[0,1]:.4f}, {actions_cpu_np[0,2]:.4f}, ...]"
        )
        logger.info(f"TT output:  [{actions_tt_np[0,0]:.4f}, {actions_tt_np[0,1]:.4f}, {actions_tt_np[0,2]:.4f}, ...]")

        # Assert PCC threshold
        PCC_THRESHOLD = 0.90
        assert pcc >= PCC_THRESHOLD, f"PCC {pcc:.4f} below threshold {PCC_THRESHOLD}"

        logger.info("SmolVLA PCC test PASSED")

    except Exception as e:
        logger.error(f"SmolVLA PCC test failed: {e}")
        raise
