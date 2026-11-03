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
from models.demos.minimax_m3.reference.vision import MiniMaxM3VisionConfig, rope_cos_sin
from models.demos.minimax_m3.tt.vision.common import to_device
from models.tt_transformers.tt.rope import RotaryEmbedding


def rope_cos_sin_padded(
    grid_thw: torch.Tensor, head_dim: int, padded_head_dim: int, theta: float, spatial_merge_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """cos and sin (L, padded_head_dim) in fp32, in the head order of q and k on device.

    The reference's columns are the angles of the 39 rotated pairs, then the same 39 again; lanes 2i and 2i+1 take
    angle i. The pass-through and pad lanes take cos 1 and sin 0, which leaves them as they are.
    """
    cos, sin = rope_cos_sin(grid_thw, head_dim, theta, spatial_merge_size)  # (L, 78)
    num_patches, rot_dim = cos.shape
    cos_padded = torch.ones(num_patches, padded_head_dim, dtype=cos.dtype)
    sin_padded = torch.zeros(num_patches, padded_head_dim, dtype=sin.dtype)
    # Each angle twice in a row, (1, 1, L, 78).
    cos_pairs, sin_pairs = RotaryEmbedding.permute_to_meta_format(cos, sin)
    cos_padded[:, :rot_dim] = cos_pairs[0, 0]
    sin_padded[:, :rot_dim] = sin_pairs[0, 0]
    return cos_padded, sin_padded


def rope_tensors(
    mesh_device, grid_thw: torch.Tensor, config: MiniMaxM3VisionConfig, padded_head_dim: int
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """The tables of rope_cos_sin_padded on mesh_device for the head dim, RoPE theta and merge size of config, new
    tensors that the caller owns.

    Shaped (1, 1, L, padded_head_dim) so the op broadcasts them over the heads, and bfloat16, the only dtype
    rotary_embedding_llama takes.
    """
    cos, sin = rope_cos_sin_padded(
        grid_thw, config.head_dim, padded_head_dim, config.rope_theta, config.spatial_merge_size
    )
    return tuple(to_device(table.view(1, 1, *table.shape), mesh_device) for table in (cos, sin))
