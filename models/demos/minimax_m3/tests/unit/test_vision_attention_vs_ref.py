# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Accuracy test for M3VLAttention (packed QKV, 3D RoPE, full SDPA) vs Hugging Face's MiniMaxM3VLVisionAttention.

Runs layer 0's self-attention on the input it gets inside the reference tower (the layer_norm1 output), with the cos
and sin that the tower gives its layers; the device's RoPE tables are computed on host from the image grid, as the
model does. Random weights and pixel values (tests/conftest.py), no checkpoint.

Run:
    pytest models/demos/minimax_m3/tests/unit/test_vision_attention_vs_ref.py
"""
from __future__ import annotations

import torch

import ttnn
from models.common.tensor_utils import nearest_32
from models.demos.minimax_m3.tt.vision.attention import M3VLAttention
from models.demos.minimax_m3.tt.vision.rope import rope_tensors

from ..test_factory import parametrize_mesh_with_fabric
from .vision_helpers import (
    ATTENTION_CI_GRIDS,
    assert_close,
    grid_name,
    max_rel_l2,
    parametrize_vision_grids,
    rows_to_device,
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
    attention = reference.tower.layers[0].self_attn
    ref_out, _ = attention(x_in[None], position_embeddings=acts["rope"])  # the output and the attention weights
    ref_out = ref_out[0]

    padded_head_dim = nearest_32(config.head_dim)  # as M3VLVisionModel pads the heads
    tt_attn = M3VLAttention.from_torch(mesh_device, attention, padded_head_dim, dtype=ttnn.bfloat16)
    cos_tt, sin_tt = rope_tensors(mesh_device, acts["grid_thw"], config, padded_head_dim)
    out = to_host(tt_attn(rows_to_device(x_in, mesh_device), cos_tt, sin_tt), mesh_device, ref_out.shape)

    assert_close(
        f"attention {grid_name(grid_thw)} L={x_in.shape[0]}", ref_out, out, 0.9999, max_rel_l2("attention", grid_thw)
    )
