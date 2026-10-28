# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Audio projections with explicit 2D-multicast program configs (E5c).

Same precision as the functional audio path: bf16 weights, HiFi4 with fp32 destination
accumulation (the caller's compute kernel config). Only the work split changes: the default
config ran the 1024 -> 4096 / 4096 -> 1024 / q / k / v projections on 88-104 cores with
in0_block_w 1; here a 2D multicast config over the full grid with in0_block_w up to 8.

`audio_linear(x, w, ...)` takes x [B, 1, S, K] (S a multiple of 32), folds the batch into M (a view
for tile-aligned S) and optionally fuses a unary activation into the matmul (e.g. the
ClippableLinear output clamp as HARDTANH). The config depends on the shapes only and is cached.
"""

from __future__ import annotations

from functools import lru_cache

import ttnn

# Static circular-buffer budget per core (Blackhole L1 is 1.5 MB; the allocator and other
# program buffers need the rest), as in tt/linear.py.
_CB_BUDGET_BYTES = 900 * 1024
_TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.float32: 4096}


def _largest_divisor(n: int, limit: int) -> int:
    return max(d for d in range(1, limit + 1) if n % d == 0)


def _cb_bytes(pm: int, pn: int, ibw: int, in0_tile: int, w_tile: int, out_tile: int) -> int:
    return 2 * pm * ibw * in0_tile + 2 * ibw * pn * w_tile + pm * pn * out_tile + pm * pn * 4096


@lru_cache(maxsize=None)
def mcast2d_config(M: int, K: int, N: int, gx: int, gy: int, in0_dtype, out_dtype, activation=None):
    """2D multicast config (fp32 destination: at most 4 tiles per output subblock)."""
    Mt, Kt, Nt = M // 32, K // 32, N // 32
    gy = min(gy, Mt)
    pm = -(-Mt // gy)
    pn = -(-Nt // gx)
    ibw = _largest_divisor(Kt, 8)
    while _cb_bytes(pm, pn, ibw, _TILE_BYTES[in0_dtype], 2048, _TILE_BYTES[out_dtype]) > _CB_BUDGET_BYTES:
        if ibw == 1:
            return None
        ibw = _largest_divisor(Kt, ibw - 1)
    sbw = _largest_divisor(pn, 4)
    sbh = max(d for d in range(1, 5) if pm % d == 0 and d * sbw <= 4)
    act = None if activation is None else ttnn.UnaryWithParam(activation[0], *activation[1:])
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=ibw,
        out_subblock_h=sbh,
        out_subblock_w=sbw,
        per_core_M=pm,
        per_core_N=pn,
        transpose_mcast=False,
        fused_activation=act,
    )


def hardtanh(lo: float, hi: float) -> tuple:
    """Fused clamp(x, lo, hi) for `audio_linear(activation=...)` (hashable for the config cache)."""
    return (ttnn.UnaryOpType.HARDTANH, float(lo), float(hi))


def audio_linear(
    x: ttnn.Tensor,
    w: ttnn.Tensor,
    *,
    dtype,
    compute_kernel_config,
    activation: tuple | None = None,
    bias: ttnn.Tensor | None = None,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """x [B, 1, S, K] @ w [1, 1, K, N] (+ bias) -> [B, 1, S, N] `dtype` TILE, `activation` fused."""
    shape = x.shape
    B, S, K = shape[0], shape[-2], shape[-1]
    N = w.shape[-1]
    M = B * shape[1] * S
    x2 = ttnn.reshape(x, (1, 1, M, K)) if B * shape[1] != 1 else x
    grid = x.device().compute_with_storage_grid_size()
    pc = mcast2d_config(M, K, N, grid.x, grid.y, x.dtype, dtype, activation)
    if pc is None:
        raise ValueError(f"no 2D config fits L1 for M={M} K={K} N={N}")
    out = ttnn.linear(
        x2,
        w,
        bias=bias,
        program_config=pc,
        compute_kernel_config=compute_kernel_config,
        memory_config=memory_config,
        dtype=dtype,
    )
    if x2 is not x:
        out = ttnn.reshape(out, (B, shape[1], S, N))
    return out
