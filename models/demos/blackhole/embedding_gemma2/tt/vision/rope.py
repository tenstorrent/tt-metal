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
with R a constant [64, 64] block-diagonal signed permutation. Each 32-channel block is exactly one
tile column, and R's diagonal block T = R[:32, :32] is the same for both, so this is what
`ttnn.experimental.rotary_embedding_llama` (prefill mode) computes per 32-wide tile with
`trans_mat = T`: one fused op instead of matmul + 2 multiplies + add (E5b: 44 us vs 130 us at
[1, 12, 2560, 64]; 152 vs 491 us at batch 4; same output). T is exact in bf16.

cos/sin depend on each image's patch positions, so the host builds them per forward
(`axial_cos_sin`) and uploads them as forward inputs [B, 1, S, D]. The fused op takes cos/sin with
batch 1, so for B > 1 `expand(cos, sin, num_heads)` turns them once per forward into the
head-expanded form [1, B*H, S, D], and q/k are viewed as [1, B*H, S, D] (free reshape).
Padding patches keep HF's position -1 (their rows are masked as keys and dropped by the pooler).

Contract
    forward(x, cos, sin)   x [B, H, S, D] bf16 TILE; cos/sin [B, 1, S, D] or expand() output
                           -> [B, H, S, D] bf16 TILE DRAM
"""

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


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
    def __init__(self, head_dim: int, trans_mat: ttnn.Tensor):
        super().__init__()
        self.head_dim = head_dim
        self.trans_mat = trans_mat
        self.compute_kernel_config = ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    @classmethod
    def build(cls, head_dim: int, device) -> "AxialRope":
        tile = ttnn.TILE_SIZE
        if head_dim != 2 * tile:
            raise ValueError(f"fused axial RoPE needs head_dim == {2 * tile} (one tile per spatial half)")
        T = rotate_half_blocks_matrix(head_dim)[:tile, :tile].reshape(1, 1, tile, tile)
        trans_mat = ttnn.from_torch(
            T, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        return cls(head_dim, trans_mat)

    @staticmethod
    def expand(cos: ttnn.Tensor, sin: ttnn.Tensor, num_heads: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """[B, 1, S, D] tables -> the form `forward` consumes without per-call work.

        B == 1: unchanged. B > 1: [1, B*H, S, D] (each image's table repeated over its heads).
        """
        B, _, S, D = cos.shape
        if B == 1:
            return cos, sin
        dram = ttnn.DRAM_MEMORY_CONFIG
        out = []
        for t in (cos, sin):
            r = ttnn.repeat(t, ttnn.Shape([1, num_heads, 1, 1]), memory_config=dram)
            out.append(ttnn.reshape(r, (1, B * num_heads, S, D)))
        return out[0], out[1]

    def forward(self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor, memory_config=None) -> ttnn.Tensor:
        B, H, S, D = x.shape
        expanded = None
        if B > 1 and cos.shape[0] != 1:
            # raw [B, 1, S, D] tables (single-module tests); VisionModel passes expand() output
            cos, sin = expanded = self.expand(cos, sin, H)
        xf = ttnn.reshape(x, (1, B * H, S, D)) if B > 1 else x
        out = ttnn.experimental.rotary_embedding_llama(
            xf,
            cos,
            sin,
            self.trans_mat,
            memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        if expanded is not None:
            ttnn.deallocate(expanded[0])
            ttnn.deallocate(expanded[1])
        return ttnn.reshape(out, (B, H, S, D)) if B > 1 else out
