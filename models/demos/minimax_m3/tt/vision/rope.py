# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
MiniMax-M3 3D vision RoPE: the cos and sin tables of each image grid, computed on the host.

The angles are Hugging Face's (MiniMaxM3VL3DRotaryEmbedding). On the chip the tables follow the lane order of q and k,
which attention.py explains.
"""
from __future__ import annotations

from typing import Tuple

import torch

import ttnn
from models.demos.minimax_m3.tt.vision.common import to_device
from models.demos.minimax_m3.tt.vision.config import MiniMaxM3VisionConfig
from models.ttt_compat.tt.rope import RotaryEmbedding


def rope_axis_dim(head_dim: int) -> int:
    """Rotary dims per axis (T, H and W): 2 * (head_dim // 2) split in three, rounded down to even (26 for 80)."""
    return 2 * ((2 * (head_dim // 2) // 3) // 2)


def rope_positions(grid_thw: torch.Tensor, spatial_merge_size: int) -> torch.Tensor:
    """The (t, h, w) coordinates of every patch, (L, 3) fp32, for a (num_images, 3) grid of patches.

    The patches are in the image processor's order: each spatial_merge_size x spatial_merge_size block is consecutive.
    """
    m = spatial_merge_size
    coords = []
    for t, h, w in grid_thw.tolist():
        hi = torch.arange(h).unsqueeze(1).expand(-1, w)
        hi = hi.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
        wi = torch.arange(w).unsqueeze(0).expand(h, -1)
        wi = wi.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
        ti = torch.arange(t).repeat_interleave(h * w)
        coords.append(torch.stack([ti, hi.repeat(t), wi.repeat(t)], dim=-1))
    return torch.cat(coords).to(torch.float32)


def rope_cos_sin(
    grid_thw: torch.Tensor, head_dim: int, theta: float, spatial_merge_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """cos and sin of the 3D RoPE, each (L, 3 * rope_axis_dim(head_dim)) in fp32 (78 columns for head_dim 80).

    The columns are the T, H and W frequency bands, then the same three again, to pair with rotate_half.
    """
    axis_dim = rope_axis_dim(head_dim)
    positions = rope_positions(grid_thw, spatial_merge_size)
    inv_freq = 1.0 / (theta ** (torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim))
    freqs = torch.cat([positions[:, i : i + 1] * inv_freq for i in range(3)], dim=-1)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rope_cos_sin_padded(
    grid_thw: torch.Tensor, head_dim: int, padded_head_dim: int, theta: float, spatial_merge_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """cos and sin (L, padded_head_dim) in fp32, in the lane order of q and k on the chip.

    Lanes 2i and 2i + 1 get angle i. The lanes that do not rotate, and the padding, get cos 1 and sin 0.
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
    """Upload rope_cos_sin_padded's tables for config, as new tensors that the caller owns.

    They are bf16, the only dtype the RoPE op takes, and (1, 1, L, padded_head_dim) so the op broadcasts them over the
    heads.
    """
    cos, sin = rope_cos_sin_padded(
        grid_thw, config.head_dim, padded_head_dim, config.rope_theta, config.spatial_merge_size
    )
    return tuple(to_device(table.view(1, 1, *table.shape), mesh_device) for table in (cos, sin))
