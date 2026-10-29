# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
MiniMax-M3-VL 3D vision RoPE: the cos and sin tables of the device's RoPE op.

The angles are the reference's (reference/vision.py, rope_cos_sin), computed on host from `image_grid_thw`. On device,
`ttnn.experimental.rotary_embedding_llama` rotates q and k, one op each. It rotates each pair of lanes (2i, 2i+1) of a
head, where the reference's rotate_half rotates dims i and i + 39 of the 78 dims that rotate, so the q and k rows of
wqkv are reordered at load (attention.py) and the tables follow the same head order.
"""
from __future__ import annotations

from typing import Tuple

import torch

import ttnn
from models.demos.minimax_m3.reference.vision import rope_cos_sin
from models.demos.minimax_m3.tt.vision.common import mesh_mapper


def rope_cos_sin_padded(
    grid_thw: torch.Tensor, head_dim: int, padded_head_dim: int, theta: float, spatial_merge_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """cos and sin (L, padded_head_dim) in fp32, in the head order of q and k on device.

    The reference's columns are the angles of the 39 rotated pairs, then the same 39 again; lanes 2i and 2i+1 take
    angle i. The pass-through and pad lanes take cos 1 and sin 0, which leaves them as they are.
    """
    cos, sin = rope_cos_sin(grid_thw, head_dim, theta, spatial_merge_size)  # (L, 78)
    num_patches, rot_dim = cos.shape
    half = rot_dim // 2
    cos_padded = torch.ones(num_patches, padded_head_dim, dtype=cos.dtype)
    sin_padded = torch.zeros(num_patches, padded_head_dim, dtype=sin.dtype)
    # Each angle twice in a row, as tt_transformers' RotaryEmbedding.permute_to_meta_format lays them out.
    # repeat_interleave(2) gives the same tables but takes more host time.
    cos_padded[:, :rot_dim] = torch.stack((cos[:, :half], cos[:, :half]), dim=-1).flatten(-2)
    sin_padded[:, :rot_dim] = torch.stack((sin[:, :half], sin[:, :half]), dim=-1).flatten(-2)
    return cos_padded, sin_padded


def rope_tensors(
    mesh_device, grid_thw: torch.Tensor, head_dim: int, padded_head_dim: int, theta: float, spatial_merge_size: int
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """The tables of rope_cos_sin_padded on mesh_device, new tensors that the caller owns.

    Shaped (1, 1, L, padded_head_dim) so the op broadcasts them over the heads, and bfloat16, the only dtype
    rotary_embedding_llama takes.
    """
    cos, sin = rope_cos_sin_padded(grid_thw, head_dim, padded_head_dim, theta, spatial_merge_size)
    mm = mesh_mapper(mesh_device)
    return tuple(
        ttnn.from_torch(
            table.view(1, 1, *table.shape).to(torch.bfloat16),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mm,
        )
        for table in (cos, sin)
    )
