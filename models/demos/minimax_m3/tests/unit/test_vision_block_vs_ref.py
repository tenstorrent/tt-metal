# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
PCC test for M3VLBlock (one full encoder layer) vs the Torch reference.

Runs layer 0 on the input it gets inside the reference tower (the pre_layrnorm output), with the RoPE
cos and sin computed on host from the image grid. Random weights and pixel values (conftest.py), no
checkpoint.

Run:
    pytest models/demos/minimax_m3/tests/unit/test_vision_block_vs_ref.py
"""
from __future__ import annotations

import torch

import ttnn
from models.demos.minimax_m3.tt.vision.block import M3VLBlock
from models.demos.minimax_m3.tt.vision.rope import rope_cos_sin_padded

from ..test_factory import parametrize_mesh_with_fabric
from .vision_helpers import assert_pcc, grid_name, parametrize_vision_grids, to_device, to_host


@torch.no_grad()
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@parametrize_vision_grids()
def test_vision_block_vs_ref(mesh_device, device_params, grid_thw, vision_reference, vision_activations):
    """Layer 0 (attention and MLP, both with residual) matches the reference at PCC >= 0.99."""
    acts = vision_activations(grid_thw)
    x_in, ref_out = acts["layer0.in"], acts["layer0.out"]
    reference = vision_reference()
    config = reference.config

    tt_block = M3VLBlock.from_torch(
        mesh_device,
        reference.layers[0],
        hidden_size=config.hidden_size,
        num_heads=config.num_attention_heads,
        head_dim=config.head_dim,
        dtype=ttnn.bfloat16,
    )
    cos, sin = rope_cos_sin_padded(
        acts["grid_thw"],
        head_dim=config.head_dim,
        padded_head_dim=tt_block.attention.padded_head_dim,
        theta=config.rope_theta,
        spatial_merge_size=config.spatial_merge_size,
    )
    cos_tt, sin_tt = tt_block.attention.stage_cos_sin(cos, sin)
    out = to_host(tt_block(to_device(x_in, mesh_device), cos_tt, sin_tt), mesh_device, ref_out.shape)

    assert_pcc(f"block {grid_name(grid_thw)} L={x_in.shape[0]}", ref_out, out, 0.99)
