# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC tests for DINOv2-ViT-S backbone in RF-DETR Medium.
Tests the backbone stage against PyTorch reference.

Architecture:
  - 12 layers, hidden=384, heads=6, patch_size=16
  - No register tokens (dinov2_windowed_small)
  - Windowed: [B*4, 325, 384] (1 CLS + 324 patches per window)
  - Full attention at layers {3,6,9}: reshape to [B, 1300, 384]
  - Features at stages [3,6,9,12] → spatial [B, 384, 36, 36]
"""

import pytest
import torch

import ttnn

from models.experimental.rfdetr_medium.common import (
    RFDETR_MEDIUM_L1_SMALL_SIZE,
    TOKENS_PER_WINDOW,
    NUM_WINDOWS_SQUARED,
    VIT_HIDDEN_SIZE,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": RFDETR_MEDIUM_L1_SMALL_SIZE}],
    indirect=True,
)
def test_backbone_feature_extraction(device, torch_model, sample_image, reference_outputs):
    """
    Test backbone + projector produces features matching PyTorch reference.

    The PyTorch Joiner runs DINOv2 + projector together, outputting [B, 256, 36, 36].
    We test our TTNN backbone (4 raw features) + projector → [B, 256, 36, 36] to match.
    """
    from models.experimental.rfdetr_medium.tt.model_preprocessing import load_backbone_weights, load_projector_weights
    from models.experimental.rfdetr_medium.tt.tt_backbone import dinov2_backbone
    from models.experimental.rfdetr_medium.tt.tt_projector import projector_forward

    backbone_params = load_backbone_weights(torch_model, device)
    projector_params = load_projector_weights(torch_model, device)

    # Preprocess image: NCHW → NHWC, pad channels 3→4
    img = sample_image.permute(0, 2, 3, 1)
    img = torch.nn.functional.pad(img, (0, 1, 0, 0, 0, 0, 0, 0))
    img_tt = ttnn.from_torch(img, dtype=ttnn.bfloat16, device=device)

    # Run TTNN backbone → 4 × [B, 384, 36, 36]
    feature_maps = dinov2_backbone(img_tt, backbone_params, batch_size=1)
    assert len(feature_maps) == 4, f"Expected 4 raw backbone features, got {len(feature_maps)}"

    # Run projector → [B, 256, 36, 36]
    projected = projector_forward(feature_maps, projector_params, batch_size=1, device=device)

    # Reference srcs are post-projector [B, 256, 36, 36] (from Joiner)
    ref_srcs = reference_outputs["srcs"]
    assert len(projected) == len(ref_srcs), f"Expected {len(ref_srcs)} projected features, got {len(projected)}"

    for i, (proj, ref_src) in enumerate(zip(projected, ref_srcs)):
        proj_torch = ttnn.to_torch(proj).float() if isinstance(proj, ttnn.Tensor) else proj
        assert (
            proj_torch.shape == ref_src.shape
        ), f"Projected feature {i}: expected {ref_src.shape}, got {proj_torch.shape}"
        pcc = assert_with_pcc(ref_src, proj_torch, pcc=0.70)
        print(f"Backbone+Projector feature {i}: PCC = {pcc}")


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": RFDETR_MEDIUM_L1_SMALL_SIZE}],
    indirect=True,
)
def test_backbone_windowed_attention(device, torch_model, sample_image):
    """
    Test a single windowed DINOv2 attention block for PCC validation.
    Uses random input in windowed shape [B*4, 325, 384].
    """
    from models.experimental.rfdetr_medium.tt.model_preprocessing import load_backbone_weights
    from models.experimental.rfdetr_medium.tt.tt_backbone import dinov2_attention

    backbone_params = load_backbone_weights(torch_model, device)

    torch.manual_seed(0)
    # Windowed shape: [B*4, 325, 384] (1 CLS + 324 patches per window)
    hidden_states = torch.randn(NUM_WINDOWS_SQUARED, TOKENS_PER_WINDOW, VIT_HIDDEN_SIZE)

    # Get layer 0 reference (layer 0 is windowed)
    dinov2_backbone_model = torch_model.backbone[0].encoder.encoder
    with torch.no_grad():
        layer = dinov2_backbone_model.encoder.layer[0]
        normed = layer.norm1(hidden_states)
        attn_out = layer.attention(normed)[0]
        ls1 = layer.layer_scale1(attn_out) if hasattr(layer, "layer_scale1") else attn_out
        ref_output = hidden_states + ls1

    # Run TTNN attention
    hs_tt = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = dinov2_attention(hs_tt, backbone_params["layers"][0])
    tt_output_torch = ttnn.to_torch(tt_output).float()

    assert_with_pcc(ref_output, tt_output_torch, pcc=0.98)
