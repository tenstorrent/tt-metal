# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Explicit matmul program configs for Chronos per-token linears.

The DRAM-interleaved auto-config does not fold a batched activation's batch
into M, so ``(B, T, d) @ (d, n)`` runs as B tiny matmuls on part of the grid.
These 2D multicast configs set ``fuse_batch=True`` (M = B * padded T) and tile
the whole compute grid.
"""

from __future__ import annotations

import math

TILE = 32
# in0 + in1 double-buffered, plus the output block, in bf16 tiles (2 KiB each).
_L1_TILE_BUDGET = 560
_MAX_IN0_BLOCK_W = 12
_MAX_OUT_BLOCK_H = 16


def compute_kernel_config(math_fidelity=None, *, fp32_dest_acc_en: bool = False, packer_l1_acc: bool = True):
    import ttnn

    from models.common.utility_functions import is_blackhole

    cls = ttnn.types.BlackholeComputeKernelConfig if is_blackhole() else ttnn.WormholeComputeKernelConfig
    return cls(
        math_fidelity=ttnn.MathFidelity.HiFi2 if math_fidelity is None else math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=packer_l1_acc,
    )


def _largest_divisor_at_most(n: int, cap: int) -> int:
    return max(d for d in range(1, min(n, cap) + 1) if n % d == 0)


def _subblock(block_h: int, block_w: int, max_tiles: int) -> tuple[int, int]:
    best = (1, 1)
    for h in range(1, max_tiles + 1):
        for w in range(1, max_tiles // h + 1):
            if block_h % h or block_w % w:
                continue
            if h * w > best[0] * best[1] or (h * w == best[0] * best[1] and w > best[1]):
                best = (h, w)
    return best


def fused_batch_matmul_config(
    grid,
    m_tiles: int,
    k_tiles: int,
    n_tiles: int,
    *,
    fused_activation=None,
    fp32_dest_acc_en: bool = False,
):
    """2D mcast config with M split over grid rows and N over grid columns."""
    import ttnn

    grid_x, grid_y = grid
    per_core_n = math.ceil(n_tiles / grid_x)
    per_core_m = math.ceil(m_tiles / grid_y)
    cols = math.ceil(n_tiles / per_core_n)
    rows = math.ceil(m_tiles / per_core_m)
    out_block_w = per_core_n
    out_block_h = _largest_divisor_at_most(per_core_m, _MAX_OUT_BLOCK_H)
    in0_block_w = 1
    for cand in range(min(k_tiles, _MAX_IN0_BLOCK_W), 0, -1):
        if k_tiles % cand:
            continue
        tiles = 2 * out_block_h * cand + 2 * cand * out_block_w + out_block_h * out_block_w
        if tiles <= _L1_TILE_BUDGET:
            in0_block_w = cand
            break
    sub_h, sub_w = _subblock(out_block_h, out_block_w, 4 if fp32_dest_acc_en else 8)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(cols, rows),
        in0_block_w=in0_block_w,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        out_block_h=out_block_h,
        out_block_w=out_block_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=fused_activation,
        fuse_batch=True,
    )


_MAX_SDPA_CHUNK = 256


def sdpa_chunk_sizes(q_seq_padded: int, k_seq_padded: int) -> tuple[int, int]:
    """One chunk per sequence for short time attention (seq 160: 8.0 -> 2.3 ms vs 32/32)."""
    return min(q_seq_padded, _MAX_SDPA_CHUNK), min(k_seq_padded, _MAX_SDPA_CHUNK)


def _fused_activation(name: str | None):
    import ttnn

    if name is None:
        return None
    if name == "relu":
        return ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
    raise ValueError(f"unsupported fused activation {name!r}")


def linear(x, weight, *, bias=None, activation: str | None = None, memory_config=None, dtype=None, math_fidelity=None):
    """``ttnn.linear`` with a batch-folding program config when shapes are tile-aligned.

    Falls back to the auto-config for sub-tile geometries (the dummy test model).
    """
    import ttnn

    memory_config = ttnn.DRAM_MEMORY_CONFIG if memory_config is None else memory_config
    x_shape = tuple(x.padded_shape)
    w_shape = tuple(weight.padded_shape)
    k, n = w_shape[-2], w_shape[-1]
    # K must be logically tile-aligned so no tile padding enters the reduction;
    # a padded N only produces padding columns.
    aligned = (
        x.layout == ttnn.TILE_LAYOUT
        and not x.memory_config().is_sharded()
        and not memory_config.is_sharded()
        and x.shape[-1] % TILE == 0
        and weight.shape[-1] >= TILE
        and x_shape[-1] == k
    )
    if not aligned:
        return ttnn.linear(
            x,
            weight,
            bias=bias,
            activation=activation,
            memory_config=memory_config,
            dtype=dtype,
            compute_kernel_config=None if math_fidelity is None else compute_kernel_config(math_fidelity),
        )
    m_tiles = math.prod(x_shape[:-1]) // TILE
    grid = x.device().compute_with_storage_grid_size()
    program_config = fused_batch_matmul_config(
        (grid.x, grid.y),
        m_tiles,
        k // TILE,
        n // TILE,
        fused_activation=_fused_activation(activation),
    )
    return ttnn.linear(
        x,
        weight,
        bias=bias,
        program_config=program_config,
        memory_config=memory_config,
        dtype=dtype,
        compute_kernel_config=compute_kernel_config(math_fidelity),
    )
