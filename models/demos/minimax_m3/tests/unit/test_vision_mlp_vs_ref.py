# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Accuracy test for M3VLMLP (fc1, exact GELU, fc2) vs the Torch reference, and a host check of its weight shapes.

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
from .vision_helpers import MAX_REL_L2, assert_pcc, grid_name, parametrize_vision_grids, to_device, to_host


@torch.no_grad()
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@parametrize_vision_grids()
def test_vision_mlp_vs_ref(mesh_device, device_params, grid_thw, vision_reference, vision_activations):
    """Layer 0 MLP matches the reference at PCC >= 0.9999, within its relL2 bound and with no NaN or Inf."""
    x_in = vision_activations(grid_thw)["layer0.mlp.in"].bfloat16().float()
    mlp = vision_reference().layers[0].mlp
    ref_out = mlp(x_in)

    tt_mlp = M3VLMLP.from_torch(mesh_device, mlp, dtype=ttnn.bfloat16)
    out = to_host(tt_mlp(to_device(x_in, mesh_device)), mesh_device, ref_out.shape)

    assert_pcc(f"mlp {grid_name(grid_thw)} L={x_in.shape[0]}", ref_out, out, 0.9999, MAX_REL_L2["mlp"])


def test_vision_mlp_rejects_mismatched_weights(expect_error):
    """M3VLMLP raises when fc2 does not take fc1's outputs or a bias does not fit its weight. The checks come before
    any upload, so no device."""
    fc1_weight, fc1_bias, fc2_weight, fc2_bias = torch.zeros(16, 8), torch.zeros(16), torch.zeros(8, 16), torch.zeros(8)
    for weights, message in (
        ((fc1_weight, fc1_bias, torch.zeros(8, 12), fc2_bias), "does not take the 16 outputs of fc1.weight"),
        ((fc1_weight, torch.zeros(12), fc2_weight, fc2_bias), "fc1.bias shape"),
        ((fc1_weight, fc1_bias, fc2_weight, torch.zeros(12)), "fc2.bias shape"),
    ):
        with expect_error(AssertionError, message):
            M3VLMLP(None, *weights)
