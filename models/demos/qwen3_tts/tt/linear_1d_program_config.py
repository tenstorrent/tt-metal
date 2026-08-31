# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Shared MatmulMultiCoreReuseMultiCast1DProgramConfig builder for decode (m=1) and
short-sequence prefill linears in Qwen3-TTS MLP and attention projections.

Keeping one implementation avoids drift between gate/up/down and wqkv/wo tuning.
"""

from __future__ import annotations

import math
import os

import ttnn


def _largest_divisor(n: int, max_divisor: int = 8) -> int:
    n = max(1, int(n))
    for i in range(min(max_divisor, n), 0, -1):
        if n % i == 0:
            return i
    return 1


def find_1d_mcast_grid(k: int, n: int, max_x: int, max_y: int) -> tuple[int, int]:
    """Rectangle whose core count divides ``N`` tiles and prefers ``per_core_N >= 2``.

    Avoids a 64-core 1D config on N=128 (4 tiles) that forces a 1×1 subblock.
    """
    tile = 32
    k_tiles = math.ceil(k / tile)
    n_tiles = math.ceil(n / tile)
    max_cores = max(1, max_x * max_y)
    best = 1
    # Tiny-K decode GEMMs (SE conv2 32x128x512) are faster with max N-split
    # than a fat per_core_N / larger in0_block_w on 2 cores.
    prefer_wide = k_tiles <= 4
    for cores in range(min(max_cores, n_tiles), 0, -1):
        if n_tiles % cores != 0:
            continue
        per_n = n_tiles // cores
        k_ok = k_tiles % cores == 0
        k_per = (k_tiles // cores) if k_ok else 0
        if prefer_wide:
            best = cores
            break
        if per_n >= 2 and k_ok and k_per >= 2:
            best = cores
            break
        if per_n >= 2 and k_ok and best == 1:
            best = cores
    else:
        for cores in range(min(max_cores, n_tiles), 0, -1):
            if n_tiles % cores == 0:
                best = cores
                break
    for y in range(1, max_y + 1):
        if best % y == 0:
            x = best // y
            if x <= max_x:
                return x, y
    return best, 1


def find_2d_mcast_grid(m: int, k: int, n: int, max_x: int, max_y: int) -> tuple[int, int]:
    """``grid_y`` divides M and K tiles; ``grid_x`` divides N tiles. Largest product."""
    tile = 32
    m_tiles = math.ceil(m / tile)
    k_tiles = math.ceil(k / tile)
    n_tiles = math.ceil(n / tile)
    best = (1, 1)
    best_cores = 1
    for gy in range(max_y, 0, -1):
        if m_tiles % gy or k_tiles % gy:
            continue
        for gx in range(max_x, 0, -1):
            if n_tiles % gx:
                continue
            cores = gx * gy
            if cores > best_cores:
                best = (gx, gy)
                best_cores = cores
    return best


def make_linear_2d_program_config(
    m: int,
    k: int,
    n: int,
    grid_x: int,
    grid_y: int,
    fp32_dest_acc_en: bool,
    fused_activation=None,
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    """2D-mcast program config for prefill-sized speaker TDNNs (M > 1 tile)."""
    tile = 32
    per_core_m = max(1, math.ceil(m / (tile * grid_y)))
    per_core_n = max(1, math.ceil(n / (tile * grid_x)))
    assert k % (tile * grid_y) == 0, f"K={k} must be divisible by TILE*grid_y={tile * grid_y}"
    in0_block_w = _largest_divisor(k // (tile * grid_y))
    subblock_limit = 4 if fp32_dest_acc_en else 8
    out_subblock_w = _largest_divisor(per_core_n, subblock_limit)
    out_subblock_h = 1
    for i in range(min(subblock_limit, per_core_m), 0, -1):
        if per_core_m % i == 0 and i * out_subblock_w <= subblock_limit:
            out_subblock_h = i
            break
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=fused_activation,
        fuse_batch=True,
    )


def make_linear_1d_program_config(
    m: int,
    k: int,
    n: int,
    grid_x: int,
    grid_y: int,
    fp32_dest_acc_en: bool,
    fused_activation=None,
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    """
    Build a 1D multicast matmul program config for ttnn.linear.

    Args:
        m: Logical row count along the batch*sequence dimension (use 1 for decode).
        k, n: Inner and output feature dimensions for x @ W.T.
        grid_x, grid_y: Device compute grid from ``device.compute_with_storage_grid_size()``.
        fp32_dest_acc_en: Must match the paired WormholeComputeKernelConfig flag so that
            subblock divisibility matches the kernel.
    """
    tile_h = 32
    tile_w = 32
    num_cores = max(1, grid_x * grid_y)

    per_core_m = max(1, m // tile_h)
    # Optional sweep / tuning: multiply effective K-split before ceil (see optimization plan Phase 3).
    _k_scale = float(os.environ.get("QWEN3_TTS_LINEAR_PER_CORE_K_SCALE", "1.0"))
    per_core_k = max(1, math.ceil((k / tile_w) / num_cores * _k_scale))
    per_core_n = max(1, math.ceil((n / tile_w) / num_cores))

    subblock_limit = 4 if fp32_dest_acc_en else 8
    out_subblock_w = max(i for i in range(1, subblock_limit + 1) if per_core_n % i == 0)
    out_subblock_h = max(
        i for i in range(1, subblock_limit + 1) if per_core_m % i == 0 and i * out_subblock_w <= subblock_limit
    )

    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=per_core_k,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=fused_activation,
        mcast_in0=True,
    )
