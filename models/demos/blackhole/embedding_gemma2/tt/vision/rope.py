# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""2D axial RoPE (HF `Gemma4VisionRotaryEmbedding` + `apply_multidimensional_rope`, modeling_gemma4.py:713-900).

HF math (head_dim D = 64, spatial_dim = D/2 = 32, theta 100):
    inv_freq  = theta ** -(arange(0, 32, 2) / 32)                         16 entries
    f_x, f_y  = pos_x * inv_freq, pos_y * inv_freq                        [N, 16] each
    cos       = cat(cos f_x, cos f_x, cos f_y, cos f_y)                   [N, 64]  (sin likewise)
    y[:32]    = x[:32] * cos[:32] + rotate_half_32(x[:32]) * sin[:32]     x-position half
    y[32:]    = x[32:] * cos[32:] + rotate_half_32(x[32:]) * sin[32:]     y-position half
where rotate_half_32 maps a 32-channel block [a, b] (16 + 16) to [-b, a].

Device form: both halves at once,
    y = x * cos + (x @ R) * sin
with R a constant [64, 64] block-diagonal signed permutation (exact in bf16, so x @ R is exact
with fp32 accumulation). cos/sin depend on each image's patch positions, so the host builds them
per forward (`axial_cos_sin`) and uploads them as forward inputs, broadcast over heads.
Padding patches keep HF's position -1 (their rows are masked as keys and dropped by the pooler).

Contract
    forward(x, cos, sin)   x [B, H, S, D] bf16 TILE; cos/sin [B, 1, S, D] bf16 TILE -> [B, H, S, D] bf16
"""

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.blackhole.embedding_gemma2.tt.config import compute_kernel_config_hifi4


def axial_inv_freq(theta: float, head_dim: int) -> torch.Tensor:
    spatial_dim = head_dim // 2
    return 1.0 / (theta ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim))


def axial_cos_sin(position_ids: torch.Tensor, theta: float, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Host: position_ids [B, N, 2] (x, y; -1 = padding) -> cos, sin [B, N, head_dim] fp32 (HF layout)."""
    inv_freq = axial_inv_freq(theta, head_dim)
    freqs = position_ids[..., None].float() * inv_freq  # [B, N, 2, D/4]
    fx, fy = freqs[:, :, 0], freqs[:, :, 1]
    cos = torch.cat([fx.cos(), fx.cos(), fy.cos(), fy.cos()], dim=-1)
    sin = torch.cat([fx.sin(), fx.sin(), fy.sin(), fy.sin()], dim=-1)
    return cos, sin


def rotate_half_blocks_matrix(head_dim: int, num_blocks: int = 2) -> torch.Tensor:
    """R [head_dim, head_dim] with x @ R == cat(rotate_half(block) for block in split(x, num_blocks))."""
    block = head_dim // num_blocks
    half = block // 2
    R = torch.zeros(head_dim, head_dim)
    for b in range(num_blocks):
        o = b * block
        for j in range(half):
            R[o + j + half, o + j] = -1.0  # y[o+j]      = -x[o+j+half]
            R[o + j, o + j + half] = 1.0  # y[o+j+half] =  x[o+j]
    return R


class AxialRope(LightweightModule):
    def __init__(self, head_dim: int, rot: ttnn.Tensor):
        super().__init__()
        self.head_dim = head_dim
        self.rot = rot
        self.compute_kernel_config = compute_kernel_config_hifi4()

    @classmethod
    def build(cls, head_dim: int, device) -> "AxialRope":
        R = rotate_half_blocks_matrix(head_dim).reshape(1, 1, head_dim, head_dim)
        rot = ttnn.from_torch(
            R, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        return cls(head_dim, rot)

    def forward(self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        dram = ttnn.DRAM_MEMORY_CONFIG
        xr = ttnn.matmul(x, self.rot, compute_kernel_config=self.compute_kernel_config, memory_config=dram)
        a = ttnn.multiply(x, cos, memory_config=dram)
        b = ttnn.multiply(xr, sin, memory_config=dram)
        ttnn.deallocate(xr)
        out = ttnn.add(a, b, memory_config=dram)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        return out
