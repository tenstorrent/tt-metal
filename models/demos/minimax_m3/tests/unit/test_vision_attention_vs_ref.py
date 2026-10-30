# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Accuracy test for M3VLAttention (packed QKV, 3D RoPE, full SDPA) vs the Torch reference.

Runs layer 0's self-attention on the input it gets inside the reference tower (the layer_norm1 output).
The RoPE cos and sin are computed on host from the image grid, as the tower does. Random weights and
pixel values (conftest.py), no checkpoint.

Run:
    pytest models/demos/minimax_m3/tests/unit/test_vision_attention_vs_ref.py
"""
from __future__ import annotations

import torch

import ttnn
from models.demos.minimax_m3.tt.vision.attention import M3VLAttention
from models.demos.minimax_m3.tt.vision.rope import rope_tensors

from ..test_factory import parametrize_mesh_with_fabric
from .vision_helpers import (
    ATTENTION_CI_GRIDS,
    MAX_REL_L2,
    assert_pcc,
    grid_name,
    parametrize_vision_grids,
    to_device,
    to_host,
)


@torch.no_grad()
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@parametrize_vision_grids(ATTENTION_CI_GRIDS)
def test_vision_attention_vs_ref(mesh_device, device_params, grid_thw, vision_reference, vision_activations):
    """Layer 0 self-attention matches the reference at PCC >= 0.9999, within its relL2 bound and with no NaN or Inf."""
    acts = vision_activations(grid_thw)
    x_in = acts["layer0.self_attn.in"].bfloat16().float()
    reference = vision_reference()
    config = reference.config
    ref_out = reference.layers[0].self_attn(x_in[None], *reference.rope(acts["grid_thw"]))[0]

    tt_attn = M3VLAttention.from_torch(
        mesh_device,
        reference.layers[0].self_attn,
        hidden_size=config.hidden_size,
        num_heads=config.num_attention_heads,
        head_dim=config.head_dim,
        dtype=ttnn.bfloat16,
    )
    cos_tt, sin_tt = rope_tensors(
        mesh_device,
        acts["grid_thw"],
        head_dim=config.head_dim,
        padded_head_dim=tt_attn.padded_head_dim,
        theta=config.rope_theta,
        spatial_merge_size=config.spatial_merge_size,
    )
    out = to_host(tt_attn(to_device(x_in, mesh_device), cos_tt, sin_tt), mesh_device, ref_out.shape)

    assert_pcc(f"attention {grid_name(grid_thw)} L={x_in.shape[0]}", ref_out, out, 0.9999, MAX_REL_L2["attention"])
