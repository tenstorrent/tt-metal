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


def prefill_linear(x: ttnn.Tensor, weight: ttnn.Tensor, role: str, opts: LinearOptimizations, *, silu=False):
    """[B, S, K] x [K, N] -> [B, S, N] BF16 in DRAM.

    Long row counts use ``minimal_matmul`` (the route the Qwen3.8 prefill path measured);
    short ones use ``ttnn.linear`` with its auto-selected program config.
    """
    compute = opts.compute_kernel_cfg[role]
    rows = x.shape[-2] * (x.shape[0] if len(x.shape) == 3 else 1)
    if rows >= opts.minimal_min_rows:
        return ttnn.experimental.minimal_matmul(
            x,
            weight,
            config=opts.minimal_config,
            compute_kernel_config=compute,
            memory_config=opts.output_memcfg,
            dtype=opts.output_dtype,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if silu else None,
        )
    return ttnn.linear(
        x,
        weight,
        compute_kernel_config=compute,
        memory_config=opts.output_memcfg,
        dtype=opts.output_dtype,
        **({"activation": "silu", "core_grid": opts.core_grid} if silu else {}),
    )
