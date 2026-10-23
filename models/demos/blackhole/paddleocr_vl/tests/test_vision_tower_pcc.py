# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""S2 gate: projector-output PCC vs goldens (the projector output is order-identical to HF; the tower's is not).

Run::

    MESH_DEVICE=P150 TT_VISIBLE_DEVICES=3 \
    TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
    HF_MODEL=PaddlePaddle/PaddleOCR-VL-1.6 \
    pytest models/demos/blackhole/paddleocr_vl/tests/test_vision_tower_pcc.py
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.demos.blackhole.paddleocr_vl.tests.generate_goldens import INTERMEDIATE_SAMPLES
from models.demos.blackhole.paddleocr_vl.tt.vision.model import DropInVisionTransformer
from models.demos.blackhole.paddleocr_vl.tt.vision.vision_model_config import VisionModelArgs
from models.demos.blackhole.paddleocr_vl.tt.weight_mapping import map_vision_state_dict
from tests.ttnn.utils_for_testing import assert_with_pcc

PCC_TARGET = 0.98
# Unpadded images below the gate since bring-up; strict, so a fix shows up as XPASS.
_BELOW_GATE = pytest.mark.xfail(strict=True, reason="exact-bucket PCC below 0.98 (0.979 / 0.943)")


@pytest.fixture
def tower(device):
    model_args = VisionModelArgs(device, instruct=True, max_batch_size=1, max_seq_len=2048)
    device_sd, host_sd = map_vision_state_dict(
        model_args.load_state_dict(), vision_head_dim=model_args.head_dim, strict=True
    )
    return DropInVisionTransformer(
        model_args=model_args,
        device_state_dict=device_sd,
        host_state_dict=host_sd,
        dtype=ttnn.bfloat8_b,
    )


@pytest.mark.parametrize(
    "golden_name",
    [
        pytest.param(n, marks=_BELOW_GATE) if n in ("label_shipping", "receipt_grocery") else n
        for n in INTERMEDIATE_SAMPLES
    ],
)
def test_vision_tower_matches_projector_golden(tower, golden, golden_name):
    g = golden(golden_name)
    out = tower(g["pixel_values"].to(torch.bfloat16), g["image_grid_thw"])
    ref = g["projector_out"]

    assert out.shape == ref.shape, f"tt shape {tuple(out.shape)} != golden shape {tuple(ref.shape)}"
    assert_with_pcc(ref, out, PCC_TARGET)
