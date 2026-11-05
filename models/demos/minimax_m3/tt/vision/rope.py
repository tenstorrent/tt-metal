# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
MiniMax-M3-VL 3D vision RoPE: the angles of each patch and the cos and sin tables of the device's RoPE op.

rope_cos_sin computes the angles on host from `image_grid_thw`, as Hugging Face's MiniMaxM3VL3DRotaryEmbedding does. On
device, `ttnn.experimental.rotary_embedding_llama` rotates q and k, one op each. It rotates each pair of lanes (2i,
2i+1) of a head, where Hugging Face's rotate_half rotates dims i and i + 39 of the 78 dims that rotate, so the q and k
rows of wqkv are reordered at load (attention.py) and the tables follow the same head order.
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
    """(num_images, 3) grid of (t, h, w) patches -> (L, 3) fp32 coordinates of every patch.

    The patches are in the image processor's order, where each spatial_merge_size x spatial_merge_size
    block of patches is consecutive.
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
    """cos and sin (L, padded_head_dim) in fp32, in the head order of q and k on device.

    rope_cos_sin's columns are the angles of the 39 rotated pairs, then the same 39 again; lanes 2i and 2i+1 take
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
