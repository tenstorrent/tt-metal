# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pooler: sentence-transformers `1_Pooling` (masked mean, include_prompt) + `2_Normalize` (L2).

    pooled = sum_s keep[b, s] * h[b, s, :] / max(sum_s keep[b, s], 1)
    emb    = pooled / max(||pooled||_2, 1e-12)

Contract
    input   last_hidden [B, 1, S, D] bf16 TILE DRAM (D = 768, after the 512->768 projection)
            keep        [B, 1, 1, S] bf16 TILE (1 = valid token, 0 = padding)
    output  [B, 1, 1, D] float32 TILE DRAM, unit L2 norm per row (row 0 of each batch tile is valid)
"""

from __future__ import annotations

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.blackhole.embedding_gemma2.tt.config import compute_kernel_config_hifi4


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
        keep32 = ttnn.typecast(keep, ttnn.float32)
        counts = ttnn.sum(keep32, dim=-1, keepdim=True)  # [B, 1, 1, 1]
        counts = ttnn.clip(counts, 1.0, 1.0e9)
        mean = ttnn.divide(summed, counts, memory_config=dram)
        sq = ttnn.multiply(mean, mean)
        ss = ttnn.sum(sq, dim=-1, keepdim=True)
        norm = ttnn.sqrt(ss)
        norm = ttnn.clip(norm, self.normalize_eps, 1.0e30)
        out = ttnn.divide(mean, norm, memory_config=dram)
        for t in (summed, keep32, counts, mean, sq, ss, norm):
            ttnn.deallocate(t)
        return out
