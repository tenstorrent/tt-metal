# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Audio projections with explicit 2D-multicast program configs (E5c).

Precision is the caller's: weight dtype (the tensor `w`), compute kernel config and output dtype
come from the audio precision policy (E6b; E5c: bf16 weights, HiFi4 with fp32 destination
accumulation). Only the work split is decided here: the default
config ran the 1024 -> 4096 / 4096 -> 1024 / q / k / v projections on 88-104 cores with
in0_block_w 1; here a 2D multicast config over the full grid with in0_block_w up to 8.

`audio_linear(x, w, ...)` takes x [B, 1, S, K] (S a multiple of 32), folds the batch into M (a view
for tile-aligned S) and optionally fuses a unary activation into the matmul (e.g. the
ClippableLinear output clamp as HARDTANH). The config depends on the shapes only and is cached.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import ttnn
from models.demos.embedding_gemma2.tt.audio.config import audio_act_memory_config

# Static circular-buffer budget per core (Blackhole L1 is 1.5 MB; the allocator and other
# program buffers need the rest), as in tt/linear.py.
_CB_BUDGET_BYTES = 900 * 1024
_TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.float32: 4096, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}


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


@dataclass(frozen=True)
class AudioMMSpec:
    """One tuned audio matmul config (tile units). family "2d": `MatmulMultiCoreReuseMultiCastProgramConfig`
    on `grid` (default: the device grid) with `in0_block_w`, `sb` (out_subblock_h, out_subblock_w),
    `per_core` (per_core_M, per_core_N), `transpose` (transpose_mcast: M along the grid columns); None fields
    are derived as in `mcast2d_config`. family "1d": `MatmulMultiCoreReuseMultiCast1DProgramConfig` (in0
    multicast, N split over the grid's cores, all of M per core)."""

    family: str
    grid: tuple[int, int] | None = None
    in0_block_w: int | None = None
    sb: tuple[int, int] | None = None
    per_core: tuple[int, int] | None = None
    transpose: bool = False


# Tuned configs per compute grid (E10.1, doc/prefill_towers/): AUDIO_TUNED[grid][(M, K, N)] -> AudioMMSpec
# replaces `mcast2d_config` for that matmul shape on that grid only, at batch 1 (M = the token bucket; the
# audio projections have distinct (K, N): q|k|v 1024 -> 3072, FFN 1024 -> 4096 / 4096 -> 1024, lconv start
# 1024 -> 2048, attention post and lconv end 1024 -> 1024, the lconv shifts S -> 1024 with K = S). Grids
# without an entry (the P150a's 13 x 10) keep the rule unchanged. EG2_AUDIO_MM_TUNED=0 disables the table
# (same-session control); AUDIO_MM_OVERRIDES[(M, K, N)] (in-process, empty by default) takes precedence on
# every grid (the in-model config search sets it).
#   FFN down (4096 -> 1024): in0_block_w 32 at 256 tokens (-1.3 % burst), 16 at 768 (-0.3 %); the rule's 8
#   (fewer K passes; accumulation order changes, soft-token PCC 0.99849 -> 0.99907 at 1 x 256, 0.99767 at 1 x 768)
# Every other role and family searched (transposed 2D, 1D, in0_block_w 4 / 16) lost or tied: tt-ops
# e10/search/audio_a{1,2,3,4}.jsonl, doc/prefill_towers/.
AUDIO_TUNED: dict[tuple[int, int], dict[tuple[int, int, int], AudioMMSpec]] = {
    (12, 10): {
        (256, 4096, 1024): AudioMMSpec("2d", in0_block_w=32),
        (768, 4096, 1024): AudioMMSpec("2d", in0_block_w=16),
    }
}
AUDIO_TUNED_ON = os.environ.get("EG2_AUDIO_MM_TUNED", "1") == "1"
AUDIO_MM_OVERRIDES: dict[tuple[int, int, int], AudioMMSpec] = {}


def audio_spec(M: int, K: int, N: int, grid, batch: int = 1) -> AudioMMSpec | None:
    if batch != 1:
        return None
    spec = AUDIO_MM_OVERRIDES.get((M, K, N))
    if spec is None and AUDIO_TUNED_ON:
        spec = AUDIO_TUNED.get((grid.x, grid.y), {}).get((M, K, N))
    return spec


@lru_cache(maxsize=None)
def audio_spec_config(spec: AudioMMSpec, M: int, K: int, N: int, gx: int, gy: int, activation=None):
    """Program config for `spec` (fp32 destination: at most 4 tiles per output subblock)."""
    Mt, Kt, Nt = M // 32, K // 32, N // 32
    sgx, sgy = spec.grid or (gx, gy)
    act = None if activation is None else ttnn.UnaryWithParam(activation[0], *activation[1:])
    ibw = spec.in0_block_w or _largest_divisor(Kt, 8)
    if spec.family == "1d":
        pm, pn = spec.per_core or (Mt, -(-Nt // (sgx * sgy)))
    elif spec.family == "2d":
        if spec.per_core is not None:
            pm, pn = spec.per_core
        elif spec.transpose:
            pm, pn = -(-Mt // min(sgx, Mt)), -(-Nt // sgy)
        else:
            pm, pn = -(-Mt // min(sgy, Mt)), -(-Nt // sgx)
    else:
        raise ValueError(f"unknown audio matmul family {spec.family}")
    if spec.sb is not None:
        sbh, sbw = spec.sb
    else:
        sbw = _largest_divisor(pn, 4)
        sbh = max(d for d in range(1, 5) if pm % d == 0 and d * sbw <= 4)
    if spec.family == "1d":
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(sgx, sgy),
            in0_block_w=ibw,
            out_subblock_h=sbh,
            out_subblock_w=sbw,
            per_core_M=pm,
            per_core_N=pn,
            fuse_batch=True,
            fused_activation=act,
            mcast_in0=True,
        )
    if not spec.transpose:
        sgy = min(sgy, -(-Mt // pm))
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(sgx, sgy),
        in0_block_w=ibw,
        out_subblock_h=sbh,
        out_subblock_w=sbw,
        per_core_M=pm,
        per_core_N=pn,
        transpose_mcast=spec.transpose,
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
    memory_config=None,
) -> ttnn.Tensor:
    """x [B, 1, S, K] @ w [1, 1, K, N] (+ bias) -> [B, 1, S, N] `dtype` TILE, `activation` fused.
    `memory_config` None: L1 or DRAM by the audio intermediate rule (config.audio_act_memory_config)."""
    shape = x.shape
    B, S, K = shape[0], shape[-2], shape[-1]
    N = w.shape[-1]
    M = B * shape[1] * S
    x2 = ttnn.reshape(x, (1, 1, M, K)) if B * shape[1] != 1 else x
    if memory_config is None:
        memory_config = audio_act_memory_config(x.device(), (M, N), dtype)
    grid = x.device().compute_with_storage_grid_size()
    spec = audio_spec(M, K, N, grid, batch=B * shape[1])
    if spec is not None:
        pc = audio_spec_config(spec, M, K, N, grid.x, grid.y, activation)
    else:
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
