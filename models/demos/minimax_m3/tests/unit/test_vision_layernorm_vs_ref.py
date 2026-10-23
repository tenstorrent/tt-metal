# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
PCC test for M3VLLayerNorm vs the Torch reference.

Runs layer 0's layer_norm1 on the input it gets inside the reference tower. Random weights and pixel
values (conftest.py), no checkpoint.

Run:
    pytest models/demos/minimax_m3/tests/unit/test_vision_layernorm_vs_ref.py
"""
from __future__ import annotations

import torch

import ttnn
from models.demos.minimax_m3.tt.vision.layernorm import M3VLLayerNorm

from ..test_factory import parametrize_mesh_with_fabric
from .vision_helpers import assert_pcc, grid_name, parametrize_vision_grids, to_device, to_host


@torch.no_grad()
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@parametrize_vision_grids()
def test_vision_layernorm_vs_ref(mesh_device, device_params, grid_thw, vision_reference, vision_activations):
    """Layer 0 layer_norm1 matches the reference at PCC >= 0.99."""
    acts = vision_activations(grid_thw)
    x_in, ref_out = acts["layer0.layer_norm1.in"], acts["layer0.layer_norm1.out"]

    tt_ln = M3VLLayerNorm.from_torch(mesh_device, vision_reference().layers[0].layer_norm1, dtype=ttnn.bfloat16)
    out = to_host(tt_ln(to_device(x_in, mesh_device)), mesh_device, ref_out.shape)

    assert_pcc(f"layernorm {grid_name(grid_thw)} L={x_in.shape[0]}", ref_out, out, 0.99)
