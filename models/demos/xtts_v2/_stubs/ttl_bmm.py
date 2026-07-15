# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Full-grid tt-lang (ttl) batched matmul for the conditioning attention (tt-lang rung).

tt-lang rung (GUIDELINES/11) for the memory-bound `MatmulDeviceOperation
64 x 544 x 544` — the batched context matmul `a = v @ wtᵀ` in
`q_k_v_attention_legacy` (per-batch [ch=64, T] @ [T, T], B=heads batches). Reuses
the PROVEN full-grid (8x8) ttl matmul kernel (`ttl_matmul._mm`, fp32 dest acc),
looping over the batch dim (each head is an independent 2D matmul). bf16 TILE I/O.
"""

from __future__ import annotations

import ttnn

from models.demos.xtts_v2._stubs.ttl_matmul import _mm, TILE


def bmm(a, b, device):
    """Batched matmul a[B,M,K] @ b[B,K,N] -> [B,M,N] via the ttl 2D matmul per batch.

    Falls back to ttnn.matmul when shapes are not tile-aligned (the ttl kernel
    requires M,K,N multiples of TILE)."""
    B, M, K = int(a.shape[0]), int(a.shape[1]), int(a.shape[2])
    N = int(b.shape[2])
    if any(d % TILE for d in (M, K, N)) or int(b.shape[1]) != K:
        return ttnn.matmul(a, b)

    ab = a if a.get_dtype() == ttnn.bfloat16 else ttnn.typecast(a, ttnn.bfloat16)
    bb = b if b.get_dtype() == ttnn.bfloat16 else ttnn.typecast(b, ttnn.bfloat16)
    outs = []
    for i in range(B):
        a2d = ttnn.reshape(ttnn.slice(ab, [i, 0, 0], [i + 1, M, K]), (M, K))
        b2d = ttnn.reshape(ttnn.slice(bb, [i, 0, 0], [i + 1, K, N]), (K, N))
        y2d = ttnn.empty([M, N], ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
        _mm(a2d, b2d, y2d)
        outs.append(ttnn.reshape(y2d, (1, M, N)))
    return ttnn.concat(outs, dim=0)
