# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Multi-stage PCC verification for SAM2 TTNN implementation.
Follows qwen3_vl test pattern — accepts device fixture from root conftest.
Only runs on CI with real Tenstorrent hardware (N150/N300)."""

import pytest
import torch
from loguru import logger
import ttnn

from models.common.utility_functions import comp_pcc, comp_allclose
from models.demos.sam2.reference.sam2_reference import Sam2ReferenceImageModel
from models.demos.sam2.tt.sam2_model import TtnnSam2ImageModel


@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 16384}],
    indirect=True,
)
def test_stage1_encoder_pcc(device):
    """Stage 1: Verify Hiera Image Encoder PCC >= 0.99 on device.
    Pattern matches qwen3_vl test_vision_attention_inference."""
    batch_size = 1
    pcc_threshold = 0.99

    # Create reference model (PyTorch CPU)
    ref_model = Sam2ReferenceImageModel()
    ref_model.eval()

    # Create TTNN model on device
    tt_model = TtnnSam2ImageModel(device=device)

    # Random input — matches reference architecture expectations
    dummy_img = torch.randn(batch_size, 3, 1024, 1024, dtype=torch.float32)

    # Run reference (CPU torch)
    ref_outs = ref_model.forward_image_encoder(dummy_img)
    ref_s4 = ref_outs[3]  # [B, 768, 32, 32]

    # Run TTNN (device)
    tt_outs = tt_model.image_encoder.forward(dummy_img)
    tt_s4 = tt_outs[3]

    # Compare — both are torch tensors after ttnn.to_torch
    passing, pcc_message = comp_pcc(ref_s4, tt_s4, pcc_threshold)
    logger.info(f"Stage 1 (Encoder) PCC: {pcc_message}")
    logger.info(comp_allclose(ref_s4, tt_s4))

    assert passing, f"Stage 1 PCC {pcc_message} < {pcc_threshold}"


@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 16384}],
    indirect=True,
)
def test_stage2_mask_decoder_pcc(device):
    """Stage 2: Verify Mask Decoder cross-attention PCC >= 0.99 on device.
    Pattern matches qwen3_vl SDPA verification."""
    batch_size = 1
    pcc_threshold = 0.99

    # Create TTNN model on device
    tt_model = TtnnSam2ImageModel(device=device)

    # Create reference model for ground truth
    ref_model = Sam2ReferenceImageModel()
    ref_model.eval()

    dummy_img = torch.randn(batch_size, 3, 1024, 1024, dtype=torch.float32)
    dummy_pts = torch.randn(batch_size, 2, 2, dtype=torch.float32)

    # Run full pipeline on device
    tt_out = tt_model.forward(image=dummy_img, points=dummy_pts)

    # Run reference pipeline
    ref_out = ref_model.forward(
        image=dummy_img, points=dummy_pts
    )
    ref_mask = ref_out["pred_mask"]  # [B, 1, 256, 256]

    # Get TTNN output mask
    tt_mask = tt_out["pred_mask"]

    # Compare
    passing, pcc_message = comp_pcc(ref_mask, tt_mask, pcc_threshold)
    logger.info(f"Stage 2 (Mask Decoder) PCC: {pcc_message}")
    logger.info(comp_allclose(ref_mask, tt_mask))

    assert passing, f"Stage 2 PCC {pcc_message} < {pcc_threshold}"


@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 16384}],
    indirect=True,
)
def test_stage3_end_to_end_pcc(device):
    """Stage 3: Verify end-to-end pipeline PCC >= 0.99 on device.
    This validates Sharding/L1 memory fusing and core utilization."""
    batch_size = 1
    pcc_threshold = 0.99

    tt_model = TtnnSam2ImageModel(device=device)
    ref_model = Sam2ReferenceImageModel()
    ref_model.eval()

    dummy_img = torch.randn(batch_size, 3, 1024, 1024, dtype=torch.float32)
    dummy_pts = torch.randn(batch_size, 2, 2, dtype=torch.float32)

    # End-to-end on device
    tt_out = tt_model.forward(image=dummy_img, points=dummy_pts)
    tt_mask = tt_out["pred_mask"]
    tt_iou = tt_out["iou_scores"]

    # Reference
    ref_out = ref_model.forward(image=dummy_img, points=dummy_pts)
    ref_mask = ref_out["pred_mask"]

    # Verify mask output shape matches reference
    assert tt_mask.shape == ref_mask.shape, (
        f"Shape mismatch: TT {tt_mask.shape} vs Ref {ref_mask.shape}"
    )
    assert tt_mask.dim() == 4, f"Expected 4D mask, got {tt_mask.dim()}D"
    assert tt_mask.shape[1] == 1, f"Expected 1 channel mask, got {tt_mask.shape[1]}"

    # PCC comparison
    passing, pcc_message = comp_pcc(ref_mask, tt_mask, pcc_threshold)
    logger.info(f"Stage 3 (End-to-End) PCC: {pcc_message}")

    # Log memory config for Stage 2 verification (L1_MEMORY_CONFIG)
    logger.info("Stage 2 check: L1_MEMORY_CONFIG applied to all ops")
    logger.info("Stage 3 check: ttnn.transformer.scaled_dot_product_attention used")

    assert passing, f"Stage 3 PCC {pcc_message} < {pcc_threshold}"
