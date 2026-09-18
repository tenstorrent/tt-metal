# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""S2 gate: score the vision tower against the HuggingFace reference.

Runs the real patch tensors recorded by ``generate_goldens.py`` through the
Blackhole tower and compares the projected image embeddings to what HF produced
from the same input. Those embeddings are what gets spliced into the text
stream, so this is the gate that decides whether the vision half is correct.

The comparison is on the projector output rather than the raw tower output,
because our tower deliberately emits tokens in merge-block order while HF's is
raster. The projector output is order-identical between the two by construction
(see ``test_vision_permutation.py``), which makes it the natural meeting point.

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


@pytest.fixture(scope="module")
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


@pytest.mark.parametrize("golden_name", INTERMEDIATE_SAMPLES)
def test_vision_tower_matches_projector_golden(tower, golden, golden_name):
    g = golden(golden_name)
    out = tower(g["pixel_values"].to(torch.bfloat16), g["image_grid_thw"])
    ref = g["projector_out"]

    assert out.shape == ref.shape, f"tt shape {tuple(out.shape)} != golden shape {tuple(ref.shape)}"
    assert_with_pcc(ref, out, PCC_TARGET)
