# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
PCC test for M3VLMLP (fc1, exact GELU, fc2) vs the Torch reference.

Runs layer 0's MLP on the input it gets inside the reference tower. Random weights and pixel values
(conftest.py), no checkpoint.

Run:
    pytest models/demos/minimax_m3/tests/unit/test_vision_mlp_vs_ref.py
"""
from __future__ import annotations

import torch

import ttnn
from models.demos.minimax_m3.tt.vision.mlp import M3VLMLP

from ..test_factory import parametrize_mesh_with_fabric
from .vision_helpers import assert_pcc, grid_name, parametrize_vision_grids, to_device, to_host


@torch.no_grad()
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@parametrize_vision_grids()
def test_vision_mlp_vs_ref(mesh_device, device_params, grid_thw, vision_reference, vision_activations):
    """Layer 0 MLP matches the reference at PCC >= 0.99."""
    acts = vision_activations(grid_thw)
    x_in, ref_out = acts["layer0.mlp.in"], acts["layer0.mlp.out"]

    tt_mlp = M3VLMLP.from_torch(mesh_device, vision_reference().layers[0].mlp, dtype=ttnn.bfloat16)
    out = to_host(tt_mlp(to_device(x_in, mesh_device)), mesh_device, ref_out.shape)

    assert_pcc(f"mlp {grid_name(grid_thw)} L={x_in.shape[0]}", ref_out, out, 0.99)
