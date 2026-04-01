# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC tests for MultiScaleProjector in RF-DETR Medium.
Tests on-device projector (ttnn.conv2d) against PyTorch reference.

Input:  4 × [B, 384, 36, 36] NCHW (from backbone)
Output: 1 × [B, 256, 36, 36] NCHW (P4)
"""

import pytest
import torch

import ttnn

from models.experimental.rfdetr_medium.common import (
    RFDETR_MEDIUM_L1_SMALL_SIZE,
    HIDDEN_DIM,
    NUM_PATCHES_PER_SIDE,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": RFDETR_MEDIUM_L1_SMALL_SIZE}],
    indirect=True,
)
def test_projector(device, torch_model, sample_image):
    """
    Test projector fuses 4 backbone features into P4 feature map [B, 256, 36, 36].
    Uses raw DINOv2 features as input (not post-projector reference srcs).
    """
    from models.experimental.rfdetr_medium.tt.tt_projector import projector_forward
    from models.experimental.rfdetr_medium.tt.model_preprocessing import load_projector_weights

    projector_params = load_projector_weights(torch_model, device)

    # Get raw DINOv2 features [B, 384, 36, 36] from reference model
    dino = torch_model.backbone[0].encoder  # DinoV2
    with torch.no_grad():
        raw_features = dino(sample_image)  # list of 4 × [B, 384, 36, 36]

    # Convert to TTNN
    feature_maps_tt = []
    for feat in raw_features:
        feature_maps_tt.append(ttnn.from_torch(feat, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device))

    # Run TTNN projector on device
    projected = projector_forward(feature_maps_tt, projector_params, batch_size=1, device=device)

    # Verify output shape: [B, 256, 36, 36]
    assert len(projected) == 1, f"Expected 1 projected feature (P4), got {len(projected)}"
    p = projected[0]
    p_torch = ttnn.to_torch(p).float()
    assert p_torch.shape[1] == HIDDEN_DIM, f"Expected {HIDDEN_DIM} channels, got {p_torch.shape[1]}"
    assert p_torch.shape[2] == NUM_PATCHES_PER_SIDE, f"Expected H={NUM_PATCHES_PER_SIDE}, got {p_torch.shape[2]}"
    print(f"Projected feature shape: {p_torch.shape}")

    # Run PyTorch projector for PCC comparison
    projector_module = torch_model.backbone[0].projector
    with torch.no_grad():
        ref_projected = projector_module(raw_features)

    pcc = assert_with_pcc(ref_projected[0], p_torch, pcc=0.85)
    print(f"Projector PCC: {pcc}")
