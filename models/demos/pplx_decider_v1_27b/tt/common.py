# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Small shared helpers: weight resolution and the prefill projection used by every module."""

from __future__ import annotations

import ttnn
from models.common.modules.lazy_weight import LazyWeight, resolve_lazy_weight
from models.demos.pplx_decider_v1_27b.tt.optimizations import LinearOptimizations


def resolve(weight: LazyWeight, device, *, layout=ttnn.TILE_LAYOUT) -> LazyWeight:
    """Fill device placement for a weight built by the adapter (replicated, DRAM interleaved)."""
    return resolve_lazy_weight(
        weight, device=device, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper_config=None
    )


def concat_owned(parts: list, dim: int) -> ttnn.Tensor:
    """Concatenate tensors the caller owns and free them. A single part is returned as is:
    ``ttnn.concat`` of one tensor returns that tensor, so freeing the parts would free the result."""
    if len(parts) == 1:
        return parts[0]
    out = ttnn.concat(parts, dim=dim)
    for p in parts:
        ttnn.deallocate(p)
    return out


def prefill_linear(
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    role: str,
    opts: LinearOptimizations,
    *,
    activation: str | None = None,
    fuse_swiglu=False,
    regime_rows: int | None = None,
):
    """[B, S, K] x [K, N] -> [B, S, N] BF16 in DRAM.

    Long row counts use ``minimal_matmul`` with the per-shape config of ``opts.minimal_configs``.
    Short ones (at most ``opts.short_max_rows``) use ``ttnn.linear`` with the per-shape 1D
    in0-multicast config of ``opts.short_mcast1d``; any other shape (the readout) keeps the
    auto-selected program config.
    ``activation`` ("silu" or "sigmoid") runs in the matmul epilogue.
    ``fuse_swiglu`` (weight in the tile-pair interleaved gate/up layout) -> [B, S, N/2]
    ``silu(gate) * up`` from the matmul epilogue; only ``minimal_matmul`` has it, at every length.
    The same stored weight serves every length: the configs change, never the weight layout.
    A batch [B, S, K] (B > 1) runs as [1, g*S, K] matmuls over groups of g rows (views when S is tile
    aligned, as every prefill bucket is), viewed back to [B, S, N]; see ``batch_rows_per_call``.
    ``regime_rows`` (internal): choose the kernel path for this many rows instead of the actual count.
    """
    if len(x.shape) == 3 and x.shape[0] > 1:
        return _batched_linear(x, weight, role, opts, activation=activation, fuse_swiglu=fuse_swiglu)
    compute = opts.compute_kernel_cfg[role]
    rows = x.shape[-2] * (x.shape[0] if len(x.shape) == 3 else 1)
    k, n = weight.shape[-2], weight.shape[-1]
    path_rows = regime_rows or rows
    short = path_rows <= opts.short_max_rows
    m_tiles = x.padded_shape[-2] // 32 * (x.shape[0] if len(x.shape) == 3 else 1)
    if fuse_swiglu:
        swiglu_config = (
            opts.short_swiglu_config(m_tiles) if short and opts.short_swiglu_blocks else opts.minimal_config_for(k, n)
        )
        return ttnn.experimental.minimal_matmul(
            x,
            weight,
            config=swiglu_config,
            compute_kernel_config=compute,
            memory_config=opts.output_memcfg,
            dtype=opts.output_dtype,
            fuse_swiglu=True,
        )
    if path_rows >= opts.minimal_min_rows:
        return ttnn.experimental.minimal_matmul(
            x,
            weight,
            config=opts.minimal_config_for(k, n),
            compute_kernel_config=compute,
            memory_config=opts.output_memcfg,
            dtype=opts.output_dtype,
            fused_activation=ttnn.UnaryWithParam(_UNARY[activation]) if activation else None,
        )
    program_config = opts.short_program_config(k, n, m_tiles) if short and not activation else None
    if program_config is not None:
        return ttnn.linear(
            x,
            weight,
            program_config=program_config,
            compute_kernel_config=compute,
            memory_config=opts.output_memcfg,
            dtype=opts.output_dtype,
        )
    return ttnn.linear(
        x,
        weight,
        compute_kernel_config=compute,
        memory_config=opts.output_memcfg,
        dtype=opts.output_dtype,
        **({"activation": activation, "core_grid": opts.core_grid} if activation else {}),
    )


_UNARY = {"silu": ttnn.UnaryOpType.SILU, "sigmoid": ttnn.UnaryOpType.SIGMOID}

# Short-row (128 bucket) batches: rows per call of the 1D in0-multicast config, as M tiles (L1 bound).
SHORT_BATCH_MAX_M_TILES = 16


def batch_rows_per_call(s: int, b: int, k: int, n: int, opts: LinearOptimizations, *, fuse_swiglu=False) -> int:
    """Rows g of a [B, s, K] batch per matmul call, so that each call runs the kernel regime of one row.

    A row's output depends on the program a call gets, not on the other rows, as long as the K blocking
    and the core-grid orientation are those of batch 1 (measured: rows of a call are bit-identical to
    batch 1 then; the attention q|k|v shape differs when either changes). So:
    - fused SwiGLU (one K block size on every path, N never below M): the whole batch;
    - minimal_matmul rows (s >= minimal_min_rows): as many rows as keep ``M > N`` (the op transposes its
      core grid for M > N) equal to batch 1's ``s > N``;
    - short rows with a 1D in0-multicast config: up to ``SHORT_BATCH_MAX_M_TILES`` of M per call, same config;
    - short rows that batch 1 runs with ``ttnn.linear`` auto: one row per call (its config depends on M).
    """
    if fuse_swiglu:
        return b
    if s >= opts.minimal_min_rows:
        if s > n:
            return b
        return max(1, min(b, n // s))
    if s <= opts.short_max_rows and (k, n) in opts.short_mcast1d:
        return max(1, min(b, SHORT_BATCH_MAX_M_TILES // max(1, s // 32)))
    return 1


def _batched_linear(x, weight, role, opts, *, activation=None, fuse_swiglu=False):
    b, s, k = x.shape
    g = batch_rows_per_call(s, b, k, weight.shape[-1], opts, fuse_swiglu=fuse_swiglu)
    outs = []
    for start in range(0, b, g):
        rows = min(g, b - start)
        part = x if rows == b else ttnn.slice(x, [start, 0, 0], [start + rows, s, k])
        flat = ttnn.reshape(part, [1, rows * s, k])
        regime = s if rows > 1 and not fuse_swiglu else None
        out = prefill_linear(
            flat, weight, role, opts, activation=activation, fuse_swiglu=fuse_swiglu, regime_rows=regime
        )
        if part is not x:
            ttnn.deallocate(part)
        outs.append(ttnn.reshape(out, [rows, s, out.shape[-1]]))
    return concat_owned(outs, dim=0)
