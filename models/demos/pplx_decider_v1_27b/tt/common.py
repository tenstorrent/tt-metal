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


def prefill_linear(
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    role: str,
    opts: LinearOptimizations,
    *,
    activation: str | None = None,
    fuse_swiglu=False,
):
    """[B, S, K] x [K, N] -> [B, S, N] BF16 in DRAM.

    Long row counts use ``minimal_matmul`` (the route the Qwen3.8 prefill path measured);
    short ones use ``ttnn.linear`` with its auto-selected program config.
    ``activation`` ("silu" or "sigmoid") runs in the matmul epilogue.
    ``fuse_swiglu`` (weight in the tile-pair interleaved gate/up layout) -> [B, S, N/2]
    ``silu(gate) * up`` from the matmul epilogue; only ``minimal_matmul`` has it, at every length.
    """
    compute = opts.compute_kernel_cfg[role]
    rows = x.shape[-2] * (x.shape[0] if len(x.shape) == 3 else 1)
    if fuse_swiglu:
        return ttnn.experimental.minimal_matmul(
            x,
            weight,
            config=opts.minimal_config,
            compute_kernel_config=compute,
            memory_config=opts.output_memcfg,
            dtype=opts.output_dtype,
            fuse_swiglu=True,
        )
    if rows >= opts.minimal_min_rows:
        return ttnn.experimental.minimal_matmul(
            x,
            weight,
            config=opts.minimal_config,
            compute_kernel_config=compute,
            memory_config=opts.output_memcfg,
            dtype=opts.output_dtype,
            fused_activation=ttnn.UnaryWithParam(_UNARY[activation]) if activation else None,
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
