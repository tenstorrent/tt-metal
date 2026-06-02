# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
PolyNorm3 backward - TT-Lang
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import ttnn

TILE = 32


@ttl.operation(grid="auto", fp32_dest_acc_en=True)
def polynorm3_bw(x, dout, weight_strip, eps_tile, grad_x, grad_packed):
    rows = x.shape[0] // TILE
    cols = x.shape[1] // TILE
    one = (1, 1)
    n_row_elems = x.shape[1]
    inv_n_row = 1.0 / float(n_row_elems)

    grid_cols, grid_rows = ttl.grid_size(dims=2)
    # How many cores in the grid
    n_cores = grid_rows * grid_cols

    rows_per_node = -(-rows // n_cores)
    # Input buffers
    x_tile = ttl.make_dataflow_buffer_like(x, shape=one, block_count=2)
    dout_tile = ttl.make_dataflow_buffer_like(dout, shape=one, block_count=2)
    gx_accum_tile = ttl.make_dataflow_buffer_like(x, shape=one, block_count=2)
    eps_dfb = ttl.make_dataflow_buffer_like(eps_tile, shape=one, block_count=1)

    # Pass-1 strip accumulators
    inv_rms_x = ttl.make_dataflow_buffer_like(x, shape=one, block_count=2)
    inv_rms_x2 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=2)
    inv_rms_x3 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=2)

    scalar_1 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=2)
    scalar_2 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=2)
    scalar_3 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=2)

    # Weights
    w0_dfb = ttl.make_dataflow_buffer_like(weight_strip, shape=one, block_count=2)
    w1_dfb = ttl.make_dataflow_buffer_like(weight_strip, shape=one, block_count=2)
    w2_dfb = ttl.make_dataflow_buffer_like(weight_strip, shape=one, block_count=2)

    # coeff_k = inv_rms_k^3 * (scalar_k * w_k) * (1/N)
    coeff1 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    coeff2 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    coeff3 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)

    # wk * inv_rms_k
    w2_inv_rms_x = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    w1_inv_rms_x2 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    w0_inv_rms_x3 = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)

    # Output buffers
    dl_dw0_row = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    dl_dw1_row = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    dl_dw2_row = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    dl_db_row = ttl.make_dataflow_buffer_like(x, shape=one, block_count=2)

    # Helper buffers for the partial computation
    t1_scratch = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)
    red_dfb = ttl.make_dataflow_buffer_like(x, shape=one, block_count=1)

    @ttl.compute()
    def compute():
        node_col, node_row = ttl.node(dims=2)
        node_linear = node_row * grid_cols + node_col
        # Pass-1 strip accumulations
        for local_row in range(rows_per_node):
            row = node_linear * rows_per_node + local_row
            if row < rows:
                with inv_rms_x.reserve() as z0:
                    z0.store(ttl.math.fill(z0, 0.0))
                with inv_rms_x2.reserve() as z0:
                    z0.store(ttl.math.fill(z0, 0.0))
                with inv_rms_x3.reserve() as z0:
                    z0.store(ttl.math.fill(z0, 0.0))
                with scalar_1.reserve() as z0:
                    z0.store(ttl.math.fill(z0, 0.0))
                with scalar_2.reserve() as z0:
                    z0.store(ttl.math.fill(z0, 0.0))
                with scalar_3.reserve() as z0:
                    z0.store(ttl.math.fill(z0, 0.0))
                with red_dfb.reserve() as z0:
                    z0.store(ttl.math.fill(z0, 0.0))

                for _local_col in range(cols):
                    with (
                        # inputs
                        x_tile.wait() as xv,
                        dout_tile.wait() as dv,
                        # intermediate results
                        inv_rms_x.wait() as s2,
                        inv_rms_x2.wait() as s4,
                        inv_rms_x3.wait() as s6,
                        scalar_1.wait() as sxd,
                        scalar_2.wait() as sx2d,
                        scalar_3.wait() as sx3d,
                        red_dfb.wait() as sd,
                    ):
                        ninv = ttl.math.fill(xv, inv_n_row)
                        reduce_tile = ttl.math.fill(ninv, 1.0)
                        x2v = xv * xv
                        x3v = x2v * xv
                        x4v = x2v * x2v
                        x6v = x4v * x2v
                        ns2 = s2 + (x2v @ reduce_tile)
                        ns4 = s4 + (x4v @ reduce_tile)
                        ns6 = s6 + (x6v @ reduce_tile)
                        nsxd = sxd + ((xv * dv) @ reduce_tile)
                        nsx2d = sx2d + ((x2v * dv) @ reduce_tile)
                        nsx3d = sx3d + ((x3v * dv) @ reduce_tile)
                        nsd = sd + (dv @ reduce_tile)
                        with inv_rms_x.reserve() as o:
                            o.store(ns2)
                        with inv_rms_x2.reserve() as o:
                            o.store(ns4)
                        with inv_rms_x3.reserve() as o:
                            o.store(ns6)
                        with scalar_1.reserve() as o:
                            o.store(nsxd)
                        with scalar_2.reserve() as o:
                            o.store(nsx2d)
                        with scalar_3.reserve() as o:
                            o.store(nsx3d)
                        with red_dfb.reserve() as o:
                            o.store(nsd)

                with red_dfb.wait() as sd:
                    with dl_db_row.reserve() as o:
                        o.store(sd)

                with (
                    inv_rms_x.wait() as sum_x2,
                    inv_rms_x2.wait() as sum_x4,
                    inv_rms_x3.wait() as sum_x6,
                    eps_dfb.wait() as epsv,
                ):
                    with inv_rms_x.reserve() as o:
                        o.store(ttl.math.rsqrt(sum_x2 * ttl.math.fill(sum_x2, inv_n_row) + epsv))
                    with inv_rms_x2.reserve() as o:
                        o.store(ttl.math.rsqrt(sum_x4 * ttl.math.fill(sum_x4, inv_n_row) + epsv))
                    with inv_rms_x3.reserve() as o:
                        o.store(ttl.math.rsqrt(sum_x6 * ttl.math.fill(sum_x6, inv_n_row) + epsv))

                with w0_dfb.wait() as w0v, w1_dfb.wait() as w1v, w2_dfb.wait() as w2v:
                    # 1/N_row via fill(iv, inv_n_row) stored in ninv block.
                    with inv_rms_x.wait() as iv, scalar_1.wait() as sv:
                        with t1_scratch.reserve() as ninv:
                            ninv.store(ttl.math.fill(ninv, inv_n_row))
                        ninv = t1_scratch.wait()
                        with coeff1.reserve() as o:
                            o.store(iv * iv * iv * (sv * w2v) * ninv)
                        with w2_inv_rms_x.reserve() as o:
                            o.store(iv * w2v)
                        with dl_dw2_row.reserve() as o:
                            o.store(iv * sv)
                    with inv_rms_x2.wait() as iv, scalar_2.wait() as sv:
                        with t1_scratch.reserve() as ninv:
                            ninv.store(ttl.math.fill(ninv, inv_n_row))
                        ninv = t1_scratch.wait()
                        with coeff2.reserve() as o:
                            o.store(iv * iv * iv * (sv * w1v) * ninv)
                        with w1_inv_rms_x2.reserve() as o:
                            o.store(iv * w1v)
                        with dl_dw1_row.reserve() as o:
                            o.store(iv * sv)
                    with inv_rms_x3.wait() as iv, scalar_3.wait() as sv:
                        with t1_scratch.reserve() as ninv:
                            ninv.store(ttl.math.fill(ninv, inv_n_row))
                        ninv = t1_scratch.wait()
                        with coeff3.reserve() as o:
                            o.store(iv * iv * iv * (sv * w0v) * ninv)
                        with w0_inv_rms_x3.reserve() as o:
                            o.store(iv * w0v)
                        with dl_dw0_row.reserve() as o:
                            o.store(iv * sv)

                c1 = coeff1.wait()
                c2 = coeff2.wait()
                c3 = coeff3.wait()
                w0t = w0_inv_rms_x3.wait()
                w1t = w1_inv_rms_x2.wait()
                w2t = w2_inv_rms_x.wait()

                # Pass-2 - grad_x computation
                for _local_col in range(cols):
                    xv = x_tile.wait()
                    dv = dout_tile.wait()
                    x2v = xv * xv
                    x3v = x2v * xv
                    t1 = dv * w2t - xv * c1
                    with t1_scratch.reserve() as o:
                        o.store(t1)
                    t2 = (dv * w1t - x2v * c2) * (xv + xv)
                    with t1_scratch.wait() as acc, red_dfb.reserve() as o:
                        o.store(acc + t2)
                    t3 = (dv * w0t - x3v * c3) * (x2v + x2v + x2v)
                    with red_dfb.wait() as acc, gx_accum_tile.reserve() as o:
                        o.store(acc + t3)

    @ttl.datamovement()
    def dm_read():
        node_col, node_row = ttl.node(dims=2)
        node_linear = node_row * grid_cols + node_col
        for local_row in range(rows_per_node):
            row = node_linear * rows_per_node + local_row
            if row < rows:
                # Pass-1
                for local_col in range(cols):
                    with x_tile.reserve() as b:
                        ttl.copy(x[row, local_col], b).wait()
                    with dout_tile.reserve() as b:
                        ttl.copy(dout[row, local_col], b).wait()

                # Pass-2
                with eps_dfb.reserve() as b:
                    ttl.copy(eps_tile[0, 0], b).wait()
                with w0_dfb.reserve() as b:
                    ttl.copy(weight_strip[0, 2], b).wait()
                with w1_dfb.reserve() as b:
                    ttl.copy(weight_strip[0, 1], b).wait()
                with w2_dfb.reserve() as b:
                    ttl.copy(weight_strip[0, 0], b).wait()

                for local_col in range(cols):
                    with x_tile.reserve() as b:
                        ttl.copy(x[row, local_col], b).wait()
                    with dout_tile.reserve() as b:
                        ttl.copy(dout[row, local_col], b).wait()

    @ttl.datamovement()
    def dm_write():
        node_col, node_row = ttl.node(dims=2)
        node_linear = node_row * grid_cols + node_col
        for local_row in range(rows_per_node):
            row = node_linear * rows_per_node + local_row
            if row < rows:
                for local_col in range(cols):
                    with gx_accum_tile.wait() as b:
                        ttl.copy(b, grad_x[row, local_col]).wait()
                with dl_db_row.wait() as b:
                    ttl.copy(b, grad_packed[row, 0 * 4 + 3]).wait()
                with dl_dw0_row.wait() as b:
                    ttl.copy(b, grad_packed[row, 0 * 4 + 0]).wait()
                with dl_dw1_row.wait() as b:
                    ttl.copy(b, grad_packed[row, 0 * 4 + 1]).wait()
                with dl_dw2_row.wait() as b:
                    ttl.copy(b, grad_packed[row, 0 * 4 + 2]).wait()


def _to_dev_f32(dev, t: torch.Tensor):
    return ttnn.from_torch(
        t,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
