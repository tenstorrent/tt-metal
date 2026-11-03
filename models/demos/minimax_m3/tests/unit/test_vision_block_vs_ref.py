# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Accuracy test for M3VLBlock (one full encoder layer) vs the Torch reference.

Runs layer 0 on the input it gets inside the reference tower (the pre_layrnorm output), with the RoPE
cos and sin computed on host from the image grid. Random weights and pixel values (tests/conftest.py), no
checkpoint.

Run:
    pytest models/demos/minimax_m3/tests/unit/test_vision_block_vs_ref.py
"""
from __future__ import annotations

import torch

import ttnn
from models.common.tensor_utils import nearest_32
from models.demos.minimax_m3.tt.vision.block import M3VLBlock
from models.demos.minimax_m3.tt.vision.rope import rope_tensors

from ..test_factory import parametrize_mesh_with_fabric
from .vision_helpers import MAX_REL_L2, assert_close, grid_name, parametrize_vision_grids, rows_to_device, to_host


@torch.no_grad()
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@parametrize_vision_grids()
def test_vision_block_vs_ref(mesh_device, device_params, grid_thw, vision_reference, vision_activations):
    """Layer 0 (attention and MLP, both with residual) matches the reference at PCC >= 0.9999, within its relL2 bound
    and with no NaN or Inf."""
    acts = vision_activations(grid_thw)
    x_in = acts["layer0.in"].bfloat16().float()
    reference = vision_reference()
    config = reference.config
    ref_out = reference.layers[0](x_in[None], *reference.rope(acts["grid_thw"]))[0]

    padded_head_dim = nearest_32(config.head_dim)  # as M3VLVisionModel.from_torch pads the heads
    tt_block = M3VLBlock.from_torch(mesh_device, reference.layers[0], padded_head_dim, dtype=ttnn.bfloat16)
    cos_tt, sin_tt = rope_tensors(mesh_device, acts["grid_thw"], config, padded_head_dim)
    out = to_host(tt_block(rows_to_device(x_in, mesh_device), cos_tt, sin_tt), mesh_device, ref_out.shape)

    assert_close(f"block {grid_name(grid_thw)} L={x_in.shape[0]}", ref_out, out, 0.9999, MAX_REL_L2["block"])
