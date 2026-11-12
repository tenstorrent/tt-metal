# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Device test of M3VLBlock, one encoder layer, against Hugging Face's MiniMaxM3VLVisionEncoderLayer.

It runs layer 0 on its input in the reference tower. The weights are seeded random values (tests/conftest.py), so no
checkpoint is needed.
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
    """Layer 0 matches the reference at PCC >= 0.9999, within its relL2 bound and with no NaN or Inf."""
    acts = vision_activations(grid_thw)
    x_in = acts["layer0.in"].bfloat16().float()
    reference = vision_reference()
    config = reference.config
    layer = reference.tower.layers[0]
    ref_out = layer(x_in[None], attention_mask=None, position_embeddings=acts["rope"])[0]

    padded_head_dim = nearest_32(config.head_dim)  # as M3VLVisionModel pads the heads
    tt_block = M3VLBlock.from_torch(mesh_device, layer, padded_head_dim, dtype=ttnn.bfloat16)
    cos_tt, sin_tt = rope_tensors(mesh_device, acts["grid_thw"], config, padded_head_dim)
    out = to_host(tt_block(rows_to_device(x_in, mesh_device), cos_tt, sin_tt), mesh_device, ref_out.shape)

    assert_close(f"block {grid_name(grid_thw)} L={x_in.shape[0]}", ref_out, out, 0.9999, MAX_REL_L2["block"])
