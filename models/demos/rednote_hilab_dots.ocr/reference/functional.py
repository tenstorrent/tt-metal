# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Pure-PyTorch reference implementations for dots.ocr blocks.

Standalone functions, no TTNN imports. Faithful to
modeling_dots_vision.py from rednote-hilab/dots.ocr.
"""

import math
from typing import Dict

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# vision_rmsnorm
# ---------------------------------------------------------------------------
def vision_rmsnorm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMSNorm exactly as dots.ocr RMSNorm: normalize in fp32, scale after cast back."""
    out = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)).type_as(x)
    return out * weight


# ---------------------------------------------------------------------------
# vision_patch_embed
# ---------------------------------------------------------------------------
def vision_patch_embed_forward(
    x: torch.Tensor,
    state_dict: Dict[str, torch.Tensor],
    num_channels: int = 3,
    temporal_patch_size: int = 1,
    patch_size: int = 14,
    embed_dim: int = 1536,
    eps: float = 1e-5,
) -> torch.Tensor:
    """DotsPatchEmbed: flattened patches -> Conv2d(stride=patch) -> RMSNorm.

    Args:
        x: [num_patches, C*T*P*P] flattened patches.
        state_dict: proj.weight [E, C, P, P], proj.bias [E], norm.weight [E].
    Returns: [num_patches, embed_dim]
    """
    x = x.view(-1, num_channels, temporal_patch_size, patch_size, patch_size)[:, :, 0]
    x = F.conv2d(x, state_dict["proj.weight"], state_dict["proj.bias"], stride=patch_size)
    x = x.view(-1, embed_dim)
    return vision_rmsnorm_forward(x, state_dict["norm.weight"], eps)


# ---------------------------------------------------------------------------
# vision rotary helpers (host-side tables)
# ---------------------------------------------------------------------------
def vision_rot_pos_emb(
    grid_thw: torch.Tensor, head_dim: int = 128, spatial_merge_size: int = 2, theta: float = 10000.0
) -> torch.Tensor:
    """2D rotary table per patch, [seq, head_dim//4*2] (flattened h/w halves)."""
    dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    pos_ids = []
    for t, h, w in grid_thw.tolist():
        hpos = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos = (
            hpos.reshape(h // spatial_merge_size, spatial_merge_size, w // spatial_merge_size, spatial_merge_size)
            .permute(0, 2, 1, 3)
            .flatten()
        )
        wpos = torch.arange(w).unsqueeze(0).expand(h, -1)
        wpos = (
            wpos.reshape(h // spatial_merge_size, spatial_merge_size, w // spatial_merge_size, spatial_merge_size)
            .permute(0, 2, 1, 3)
            .flatten()
        )
        pos_ids.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))
    pos_ids = torch.cat(pos_ids, dim=0)
    max_grid = int(grid_thw[:, 1:].max())
    freqs = torch.outer(torch.arange(max_grid, dtype=torch.float), inv_freq)
    return freqs[pos_ids].flatten(1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_vision(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    orig = t.dtype
    t = t.float()
    cos = freqs.cos().unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = freqs.sin().unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    return ((t * cos) + (_rotate_half(t) * sin)).to(orig)


# ---------------------------------------------------------------------------
# vision_attention
# ---------------------------------------------------------------------------
def vision_attention_forward(
    x: torch.Tensor,
    state_dict: Dict[str, torch.Tensor],
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: torch.Tensor,
    num_heads: int = 12,
) -> torch.Tensor:
    """Eager MHA with fused QKV (no bias), 2D rope, block-diagonal cu_seqlens mask.

    x: [seq, dim]; state_dict: qkv.weight [3*dim, dim], proj.weight [dim, dim].
    """
    seq, dim = x.shape
    head_dim = dim // num_heads
    qkv = F.linear(x, state_dict["qkv.weight"])
    q, k, v = qkv.reshape(seq, 3, num_heads, head_dim).permute(1, 0, 2, 3).unbind(0)
    q = _apply_rotary_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
    k = _apply_rotary_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

    mask = torch.full([1, seq, seq], torch.finfo(q.dtype).min, dtype=q.dtype)
    for i in range(1, len(cu_seqlens)):
        mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0

    q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
    attn = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(head_dim) + mask
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(attn, v).transpose(0, 1).reshape(seq, -1)
    return F.linear(out, state_dict["proj.weight"])


# ---------------------------------------------------------------------------
# vision_mlp
# ---------------------------------------------------------------------------
def vision_mlp_forward(x: torch.Tensor, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """SwiGLU: fc2(silu(fc1(x)) * fc3(x)), no biases."""
    return F.linear(
        F.silu(F.linear(x, state_dict["fc1.weight"])) * F.linear(x, state_dict["fc3.weight"]), state_dict["fc2.weight"]
    )
