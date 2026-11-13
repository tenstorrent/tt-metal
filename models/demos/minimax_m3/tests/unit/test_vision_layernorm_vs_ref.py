# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Device test of M3VLLayerNorm against the LayerNorm of Hugging Face's vision tower.

It runs layer 0's layer_norm1 on its input in the reference tower, plus an offset per row. The weights are seeded
random values (tests/conftest.py), so no checkpoint is needed.
"""
from __future__ import annotations

import torch

import ttnn
from models.demos.minimax_m3.tt.vision.layernorm import M3VLLayerNorm

from ..test_factory import parametrize_mesh_with_fabric
from .vision_helpers import MAX_REL_L2, assert_close, grid_name, parametrize_vision_grids, rows_to_device, to_host


@torch.no_grad()
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@parametrize_vision_grids()
def test_vision_layernorm_vs_ref(mesh_device, device_params, grid_thw, vision_reference, vision_activations):
    """Layer 0 layer_norm1 matches the reference at PCC >= 0.9999, within its relL2 bound and with no NaN or Inf."""
    x = vision_activations(grid_thw)["layer0.in"]
    # Rows of zero mean would let a norm without the mean subtraction (RMSNorm) pass; with this offset RMSNorm gets PCC
    # about 0.93.
    offset = 0.5 * torch.randn(x.shape[0], 1, generator=torch.Generator().manual_seed(0))
    x_in = (x + offset).bfloat16().float()
    layer_norm = vision_reference().tower.layers[0].layer_norm1
    ref_out = layer_norm(x_in)

    tt_ln = M3VLLayerNorm.from_torch(mesh_device, layer_norm, dtype=ttnn.bfloat16)
    out = to_host(tt_ln(rows_to_device(x_in, mesh_device)), mesh_device, ref_out.shape)

    assert_close(f"layernorm {grid_name(grid_thw)} L={x_in.shape[0]}", ref_out, out, 0.9999, MAX_REL_L2["layernorm"])
