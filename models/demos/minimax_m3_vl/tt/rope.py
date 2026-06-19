# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
MiniMax-M3-VL 3D vision RoPE — host-side cos/sin precompute.

Like the Qwen-VL towers, the rotary cos/sin are computed on host from the
`image_grid_thw` grid and pushed to device once per forward (they are
position-only, weight-free). This module owns that precompute and a torch
reference for the rotate-half application; the *device* application of the
rotation lives in `attention.py` (it depends on the padded head layout).

Reference (`MiniMaxM3VL3DRotaryEmbedding`, transformers 5.12):

    rope_dims = 2 * (head_dim // 2)            # 80
    axis_dim  = 2 * ((rope_dims // 3) // 2)    # 26  -> 13 freqs/axis
    inv_freq  = 1 / theta ** (arange(0, axis_dim, 2) / axis_dim)   # (13,)
    # per-image coords with the spatial_merge 2x2 block reordering:
    #   hi = arange(h)[:,None].expand(h,w).reshape(h//m,m,w//m,m).permute(0,2,1,3).flatten()
    #   wi = arange(w)[None,:].expand(h,w).reshape(h//m,m,w//m,m).permute(0,2,1,3).flatten()
    #   ti = arange(t).repeat_interleave(h*w)
    #   coords = stack([ti, hi.repeat(t), wi.repeat(t)], -1)        # (L, 3)
    freqs = cat([coords[:, i:i+1] * inv_freq for i in range(3)], -1)  # (L, 39)
    emb   = cat([freqs, freqs], -1)                                   # (L, 78)
    cos, sin = emb.cos(), emb.sin()

Apply (`apply_rotary_pos_emb_vision`): only the first `rot_dim = 78` head
dims are rotated; the last 2 pass through.

    rotate_half(x) = cat(-x[..., d/2:], x[..., :d/2])     # d = 78 -> split 39/39
    q_rot = q[..., :78] * cos + rotate_half(q[..., :78]) * sin
"""
from __future__ import annotations

from typing import Tuple

import torch


def rope_axis_dim(head_dim: int) -> int:
    """Per-axis rotary dim: 2 * ((2*(head_dim//2) // 3) // 2). head_dim=80 -> 26."""
    rope_dims = 2 * (head_dim // 2)
    return 2 * ((rope_dims // 3) // 2)


def rope_position_coords(grid_thw: torch.Tensor, spatial_merge_size: int) -> torch.Tensor:
    """(num_images, 3) grid -> (L, 3) per-patch (t, h, w) coords, merge-block ordered.

    Matches `MiniMaxM3VL3DRotaryEmbedding.forward`'s coord construction exactly
    (the spatial_merge 2x2 reordering is what the image processor uses to lay
    patches out in the sequence).
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
    grid_thw: torch.Tensor,
    head_dim: int = 80,
    theta: float = 10000.0,
    spatial_merge_size: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the (L, rot_dim) cos/sin for the 3D vision RoPE.

    rot_dim = 3 * axis_dim = 78 for head_dim 80. Returns fp32 tensors.
    """
    axis_dim = rope_axis_dim(head_dim)
    coords = rope_position_coords(grid_thw, spatial_merge_size)  # (L, 3)
    inv_freq = 1.0 / (theta ** (torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim))  # (axis_dim//2,)
    freqs = torch.cat([coords[:, i : i + 1] * inv_freq for i in range(3)], dim=-1)  # (L, 3*axis_dim//2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (L, rot_dim)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """cat(-x[..., d/2:], x[..., :d/2]) — HF rotate_half convention."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = -2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Torch reference of `apply_rotary_pos_emb_vision`.

    q, k: (..., L, num_heads, head_dim). cos, sin: (L, rot_dim). Only the
    first rot_dim head dims are rotated; the tail passes through.
    """
    rot_dim = cos.shape[-1]
    cos = cos.unsqueeze(unsqueeze_dim)  # (L, 1, rot_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    def _apply(x):
        x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
        x_rot = x_rot * cos + rotate_half(x_rot) * sin
        return torch.cat([x_rot, x_pass], dim=-1)

    return _apply(q), _apply(k)
