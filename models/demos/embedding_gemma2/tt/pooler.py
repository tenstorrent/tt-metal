# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pooler: sentence-transformers `1_Pooling` (masked mean, include_prompt) + `2_Normalize` (L2).

    pooled = sum_s keep[b, s] * h[b, s, :] / max(sum_s keep[b, s], 1)
    emb    = pooled / max(||pooled||_2, 1e-12)

The device computes emb = summed / max(||summed||_2, 1e-12) with summed = keep @ h: dividing by
the positive token count does not change the L2-normalized direction, so the count (typecast, sum,
clip, divide: 4 ops plus the reduce's fill / clone) is not computed (E9.2 topology). Results equal
up to fp32 rounding; a row with no valid token gives 0 either way.

Contract
    input   last_hidden [B, 1, S, D] bf16 TILE DRAM (D = 768, after the 512->768 projection)
            keep        [B, 1, 1, S] bf16 TILE (1 = valid token, 0 = padding)
    output  [B, 1, 1, D] float32 TILE DRAM, unit L2 norm per row (row 0 of each batch tile is valid)
"""

from __future__ import annotations

import os

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.embedding_gemma2.tt.config import compute_kernel_config_hifi4

# 1 restores the stage-1 masked mean before the L2 normalize (E9.2 same-session A/B control).
POOL_MEAN_DIVIDE = os.environ.get("EG2_POOL_MEAN_DIVIDE", "0") == "1"


class Pooler(LightweightModule):
    def __init__(self, embedding_dim: int, normalize_eps: float = 1e-12):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.normalize_eps = normalize_eps
        self.compute_kernel_config = compute_kernel_config_hifi4()

    def forward(self, last_hidden: ttnn.Tensor, keep: ttnn.Tensor) -> ttnn.Tensor:
        dram = ttnn.DRAM_MEMORY_CONFIG
        summed = ttnn.matmul(
            keep,
            last_hidden,
            dtype=ttnn.float32,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=dram,
        )  # [B, 1, 1, D]
        if not POOL_MEAN_DIVIDE:
            out = l2_normalize(summed, self.normalize_eps)
            ttnn.deallocate(summed)
            return out
        # stage-1 topology (A/B control): masked mean before the L2 normalize
        keep32 = ttnn.typecast(keep, ttnn.float32)
        counts = ttnn.sum(keep32, dim=-1, keepdim=True)  # [B, 1, 1, 1]
        ttnn.deallocate(keep32)
        counts_c = ttnn.clip(counts, 1.0, 1.0e9)
        ttnn.deallocate(counts)
        mean = ttnn.divide(summed, counts_c, memory_config=dram)
        for t in (summed, counts_c):
            ttnn.deallocate(t)
        out = l2_normalize(mean, self.normalize_eps)
        ttnn.deallocate(mean)
        return out


def l2_normalize(x: ttnn.Tensor, eps: float = 1e-12) -> ttnn.Tensor:
    """x / max(||x||_2, eps) over the last dim, DRAM output; every intermediate freed."""
    dram = ttnn.DRAM_MEMORY_CONFIG
    sq = ttnn.multiply(x, x)
    ss = ttnn.sum(sq, dim=-1, keepdim=True)
    ttnn.deallocate(sq)
    norm = ttnn.sqrt(ss)
    ttnn.deallocate(ss)
    norm_c = ttnn.clip(norm, eps, 1.0e30)
    ttnn.deallocate(norm)
    out = ttnn.divide(x, norm_c, memory_config=dram)
    ttnn.deallocate(norm_c)
    return out
