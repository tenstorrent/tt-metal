# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""C++ Metalium batched matmul for the conditioning attention (cpp rung, GUIDELINES/12).

cpp rung for the memory-bound `MatmulDeviceOperation 64 x 544 x 544` — the
batched context matmul `a = v @ wtᵀ` in `q_k_v_attention_legacy`. Reuses the
PROVEN multi-core output-tile-partitioned C++ Metalium matmul
(`cpp_matmul._run_mm`, HiFi4 + fp32 dest acc, via ttnn.generic_op) looped over
the head/batch dim. bf16 TILE I/O.
"""

from __future__ import annotations

import ttnn

from models.demos.xtts_v2._stubs.cpp_matmul import _run_mm, TILE


def bmm(a, b, device):
    """Batched matmul a[B,M,K] @ b[B,K,N] -> [B,M,N] via the C++ 2D matmul per batch.

    Falls back to ttnn.matmul when shapes are not tile-aligned."""
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
        c2d = ttnn.empty([M, N], ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
        _run_mm(a2d, b2d, c2d, device)
        outs.append(ttnn.reshape(c2d, (1, M, N)))
    return ttnn.concat(outs, dim=0)
