# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC tests for RF-DETR Medium detection heads.
Tests class_embed and bbox_embed against PyTorch reference.
"""

import pytest
import torch

import ttnn

from models.experimental.rfdetr_medium.common import (
    RFDETR_MEDIUM_L1_SMALL_SIZE,
    HIDDEN_DIM,
    NUM_QUERIES,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": RFDETR_MEDIUM_L1_SMALL_SIZE}],
    indirect=True,
)
def test_class_head(device, torch_model):
    """
    Test classification head: Linear(256, 91).
    """
    from models.experimental.rfdetr_medium.tt.model_preprocessing import load_detection_head_weights
    from models.experimental.rfdetr_medium.tt.tt_detection_heads import class_head

    head_params = load_detection_head_weights(torch_model, device)

    torch.manual_seed(42)
    hs = torch.randn(1, NUM_QUERIES, HIDDEN_DIM)

    # PyTorch reference
    with torch.no_grad():
        ref_output = torch_model.class_embed(hs)

    # TTNN
    hs_tt = ttnn.from_torch(hs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = class_head(hs_tt, head_params)
    tt_output_torch = ttnn.to_torch(tt_output).float()

    assert_with_pcc(ref_output, tt_output_torch, pcc=0.99)


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": RFDETR_MEDIUM_L1_SMALL_SIZE}],
    indirect=True,
)
def test_bbox_head(device, torch_model):
    """
    Test bounding box head: MLP(256 → 256 → 256 → 4).
    """
    from models.experimental.rfdetr_medium.tt.model_preprocessing import load_detection_head_weights
    from models.experimental.rfdetr_medium.tt.tt_detection_heads import bbox_head

    head_params = load_detection_head_weights(torch_model, device)

    torch.manual_seed(42)
    hs = torch.randn(1, NUM_QUERIES, HIDDEN_DIM)

    # PyTorch reference
    with torch.no_grad():
        ref_output = torch_model.bbox_embed(hs)

    # TTNN
    hs_tt = ttnn.from_torch(hs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = bbox_head(hs_tt, head_params)
    tt_output_torch = ttnn.to_torch(tt_output).float()

    assert_with_pcc(ref_output, tt_output_torch, pcc=0.99)


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": RFDETR_MEDIUM_L1_SMALL_SIZE}],
    indirect=True,
)
def test_detection_heads_with_reparam(device, torch_model):
    """
    Test combined detection heads with bbox reparameterization.
    """
    from models.experimental.rfdetr_medium.tt.model_preprocessing import load_detection_head_weights
    from models.experimental.rfdetr_medium.tt.tt_detection_heads import detection_heads

    head_params = load_detection_head_weights(torch_model, device)

    torch.manual_seed(42)
    hs = torch.randn(1, NUM_QUERIES, HIDDEN_DIM)
    references = torch.rand(1, NUM_QUERIES, 4) * 0.5 + 0.25  # reasonable ref points

    # PyTorch reference
    with torch.no_grad():
        ref_class = torch_model.class_embed(hs)
        ref_delta = torch_model.bbox_embed(hs)
        ref_cxcy = ref_delta[..., :2] * references[..., 2:] + references[..., :2]
        ref_wh = ref_delta[..., 2:].exp() * references[..., 2:]
        ref_coord = torch.cat([ref_cxcy, ref_wh], dim=-1)

    # TTNN
    hs_tt = ttnn.from_torch(hs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ref_tt = ttnn.from_torch(references, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_class, tt_coord = detection_heads(hs_tt, ref_tt, head_params)
    tt_class_torch = ttnn.to_torch(tt_class).float()
    tt_coord_torch = ttnn.to_torch(tt_coord).float()

    assert_with_pcc(ref_class, tt_class_torch, pcc=0.99)
    assert_with_pcc(ref_coord, tt_coord_torch, pcc=0.97)
