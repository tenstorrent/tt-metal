# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Fused silu+mul tt-lang kernel for gated_rms_norm post-processing.

The gated RMS norm pattern is:
    normed = rms_norm(x, weight, eps)
    y = normed * silu(gate)

`ttnn_gated_rms_norm` dispatches 3 ops: rms_norm, silu, mul.
This module replaces the trailing silu+mul pair with a single tt-lang
elementwise kernel, taking 3 dispatches down to 2.

Full fusion (also fusing rms_norm) is left out intentionally — that
would require a per-row reduction across head_v_dim plus a broadcast,
neither of which has a precedent pattern in this codebase. The win
from collapsing 2 → 1 dispatches is worth more than the 1 → 0 from
attempting the reduction.
"""

import ttl
import ttnn


TILE = 32


def _make_silu_mul_kernel(num_tile_cols, grid_x, grid_y):
    """Build a fused silu+mul kernel.

    Args:
        num_tile_cols: number of tile-columns to process per device.
            For head-sharded gated_rms_norm with head_v_dim=128 and
            TILE=32, that's 4 tile-columns.
        grid_x, grid_y: kernel grid. Each core handles
            num_tile_cols / (grid_x * grid_y) tile-columns.

    Returns:
        A ttl.operation that, when called with (normed, gate, y_out),
        writes `y_out = normed * silu(gate)` elementwise.
    """
    num_cores = grid_x * grid_y
    if num_tile_cols % num_cores != 0:
        raise ValueError(f"num_tile_cols={num_tile_cols} not divisible by grid {grid_x}x{grid_y}")
    tiles_per_core = num_tile_cols // num_cores

    @ttl.operation(
        grid=(grid_x, grid_y),
        fp32_dest_acc_en=False,
        options="",
    )
    def silu_mul(normed, gate, y_out):
        ni = ttl.make_dataflow_buffer_like(normed, shape=(1, 1), block_count=2)
        gi = ttl.make_dataflow_buffer_like(gate, shape=(1, 1), block_count=2)
        yo = ttl.make_dataflow_buffer_like(y_out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            for _ in range(tiles_per_core):
                with ni.wait() as nn, gi.wait() as gg:
                    y = nn * ttl.math.silu(gg)
                    with yo.reserve() as yb:
                        yb.store(y)

        @ttl.datamovement()
        def dm_read():
            nx, ny = ttl.node(dims=2)
            base = (ny * grid_x + nx) * tiles_per_core
            for i in range(tiles_per_core):
                t = base + i
                with ni.reserve() as blk:
                    ttl.copy(normed[0, t], blk).wait()
                with gi.reserve() as blk:
                    ttl.copy(gate[0, t], blk).wait()

        @ttl.datamovement()
        def dm_write():
            nx, ny = ttl.node(dims=2)
            base = (ny * grid_x + nx) * tiles_per_core
            for i in range(tiles_per_core):
                t = base + i
                with yo.wait() as blk:
                    ttl.copy(blk, y_out[0, t]).wait()

    return silu_mul


# Per-shape kernel cache. Key is num_tile_cols.
_KERNEL_BY_TILES = {}


def fused_gated_rms_norm_step(x, gate, weight, eps, y_out):
    """Drop-in replacement for ttnn_gated_rms_norm with one less dispatch.

    Args:
        x: ttnn.Tensor. Any shape with last dim = head_v_dim (tile-aligned).
            Caller may pass 2D [h, d], 3D [1, h, d], or 4D [1, 1, h, d];
            the wrapper reshapes to 2D for the kernel's 2D access pattern
            and ttnn.reshape is a view (no data movement).
        gate: same shape as x. Gating signal.
        weight: ttnn.Tensor [head_v_dim] or [1, head_v_dim] replicated.
        eps: float - RMS-norm epsilon.
        y_out: ttnn.Tensor same shape as x - pre-allocated output buffer.
            Mutated in place.

    Returns:
        y_out (same tensor passed in).
    """
    # Collapse leading dims to a single "rows" dim so the kernel sees a
    # canonical 2D [rows, head_v_dim] tensor. ttnn.reshape preserves the
    # underlying buffer (and any sharded-dim distribution) since the
    # sharded dim's element count is unchanged.
    orig_shape = list(x.shape)
    last_dim = orig_shape[-1]
    if last_dim % TILE != 0:
        raise ValueError(f"last dim {last_dim} not tile-aligned (need % {TILE} == 0)")
    rows = 1
    for d in orig_shape[:-1]:
        rows *= int(d)
    if len(orig_shape) != 2:
        x_2d = ttnn.reshape(x, [rows, last_dim])
        gate_2d = ttnn.reshape(gate, [rows, last_dim])
        y_out_2d = ttnn.reshape(y_out, [rows, last_dim])
    else:
        x_2d = x
        gate_2d = gate
        y_out_2d = y_out

    # Step 1: rms_norm via existing TTNN op (unchanged).
    normed_2d = ttnn.rms_norm(x_2d, weight=weight, epsilon=eps)

    # Step 2: fused silu+mul via tt-lang kernel.
    num_tile_cols = last_dim // TILE
    # Use as many cores as we have tile-columns (bounded by 8 for safety).
    grid_x = num_tile_cols if num_tile_cols <= 8 else 8
    grid_y = 1
    while num_tile_cols % (grid_x * grid_y) != 0 and grid_x > 1:
        grid_x -= 1
    cache_key = (num_tile_cols, grid_x, grid_y)
    if cache_key not in _KERNEL_BY_TILES:
        _KERNEL_BY_TILES[cache_key] = _make_silu_mul_kernel(num_tile_cols, grid_x, grid_y)
    kernel_fn = _KERNEL_BY_TILES[cache_key]
    kernel_fn(normed_2d, gate_2d, y_out_2d)
    return y_out
