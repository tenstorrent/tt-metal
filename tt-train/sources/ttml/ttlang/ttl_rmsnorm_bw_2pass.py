# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
RMSNorm backward — 2-pass tile streaming.

**Pass 1:** per tile-column, accumulate row scalar
``scale = Sum_c (x · (gamma/rms) · dL_dout)`` via ``(contrib @ ones)`` matmul reduction

**Pass 2:** re-stream tiles and emit
``dL_dinput = (gamma/rms)·dL_dout - scale·x·(1/rms)^2·(1/C)``,
``dL_dgamma = x·(1/rms)·dL_dout``.
"""
from __future__ import annotations

import ttl
import ttnn
import torch

TILE_WIDTH = 32


def get_block_size(whole_row_width: int = 32, max_block_size: int = 4) -> int:
    """
    Same logic as tt-train/.../metal/common/program_utils.hpp get_block_size:
    """
    for block_size in range(max_block_size, 1, -1):
        if whole_row_width % block_size == 0:
            return block_size
    return 1


def make_kernel():
    bc = 2

    @ttl.operation(grid="auto", fp32_dest_acc_en=True)
    def rmsnorm_bw_2pass(
        input_t,
        gamma_t,
        rms_t,
        dL_dout_t,
        dL_dinput_out,
        dL_dgamma_comp_out,
    ):
        cols = input_t.shape[1]
        if cols % TILE_WIDTH != 0:
            raise ValueError(f"input_t width ({cols}) must be divisible by TILE_WIDTH ({TILE_WIDTH})")
        ht = input_t.shape[0] // TILE_WIDTH
        wt = cols // TILE_WIDTH
        block_size = get_block_size(wt, 2)
        blk = (1, block_size)
        tile = (1, 1)
        num_col_blocks = wt // block_size
        elem_c = wt * TILE_WIDTH
        inv_c_scalar = 1.0 / elem_c

        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_cores = grid_cols * grid_rows
        rows_per_core = -(-ht // total_cores)

        # Input buffers (width streamed in blocks of block_size tiles)
        inp_blk = ttl.make_dataflow_buffer_like(input_t, shape=blk, block_count=bc)
        gamma_blk = ttl.make_dataflow_buffer_like(gamma_t, shape=blk, block_count=bc)
        rms_tile = ttl.make_dataflow_buffer_like(rms_t, shape=tile, block_count=bc)
        dL_blk = ttl.make_dataflow_buffer_like(dL_dout_t, shape=blk, block_count=bc)

        # Output buffers
        out_da_blk = ttl.make_dataflow_buffer_like(dL_dinput_out, shape=blk, block_count=bc)
        out_dg_blk = ttl.make_dataflow_buffer_like(dL_dgamma_comp_out, shape=blk, block_count=bc)

        # Scale accumulator (row scalar)
        scale_acc = ttl.make_dataflow_buffer_like(input_t, shape=tile, block_count=bc)

        # Constant 1/C tile for rhs
        inv_c_dfb = ttl.make_dataflow_buffer_like(input_t, shape=tile, block_count=1)
        # Unity column for (1, block_size) @ (block_size, 1) row reduction (reduce_sum unsupported).
        reduce_col_dfb = ttl.make_dataflow_buffer_like(input_t, shape=(block_size, 1), block_count=1)
        # Broadcast scalars to block width once per row (reused each col block).
        recip_bc_blk = ttl.make_dataflow_buffer_like(input_t, shape=blk, block_count=1)
        scale_bc_blk = ttl.make_dataflow_buffer_like(input_t, shape=blk, block_count=1)
        inv_c_bc_blk = ttl.make_dataflow_buffer_like(input_t, shape=blk, block_count=1)

        @ttl.compute()
        def compute():
            core_col, core_row = ttl.node(dims=2)
            core_linear = core_row * grid_cols + core_col
            with inv_c_dfb.reserve() as ic:
                ic.store(ttl.math.fill(ic, inv_c_scalar))
            inv_c = inv_c_dfb.wait()

            with reduce_col_dfb.reserve() as rc:
                rc.store(ttl.math.fill(rc, 1.0))
            reduce_col = reduce_col_dfb.wait()

            for local_row in range(rows_per_core):
                row = core_linear * rows_per_core + local_row
                if row < ht:
                    with scale_acc.reserve() as z:
                        z.store(ttl.math.fill(z, 0.0))

                    # Pass 1: accumulate row scalar (one RMS tile per row).
                    with rms_tile.wait() as rmv:
                        recip = ttl.math.recip(rmv)
                        with recip_bc_blk.reserve() as rb:
                            rb.store(ttl.math.broadcast(recip, rb, dims=[1]))
                        with recip_bc_blk.wait() as recip_w:
                            for blk_idx in range(num_col_blocks):
                                with (
                                    inp_blk.wait() as iv,
                                    gamma_blk.wait() as gv,
                                    dL_blk.wait() as dlv,
                                    scale_acc.wait() as s,
                                ):
                                    gained = recip_w * gv * dlv
                                    contrib = iv * gained
                                    partial = contrib @ reduce_col
                                    with scale_acc.reserve() as o:
                                        o.store(s + partial)

                    scale = scale_acc.wait()

                    # Pass 2: emit dL_dinput and dL_dgamma
                    with rms_tile.wait() as rmv:
                        recip = ttl.math.recip(rmv)
                        with recip_bc_blk.reserve() as rb:
                            rb.store(ttl.math.broadcast(recip, rb, dims=[1]))
                        with scale_bc_blk.reserve() as sb:
                            sb.store(ttl.math.broadcast(scale, sb, dims=[1]))
                        with inv_c_bc_blk.reserve() as ib:
                            ib.store(ttl.math.broadcast(inv_c, ib, dims=[1]))
                        with (
                            recip_bc_blk.wait() as recip_w,
                            scale_bc_blk.wait() as scale_w,
                            inv_c_bc_blk.wait() as inv_c_w,
                        ):
                            for blk_idx in range(num_col_blocks):
                                with (
                                    inp_blk.wait() as iv,
                                    gamma_blk.wait() as gv,
                                    dL_blk.wait() as dlv,
                                ):
                                    with out_da_blk.reserve() as oa:
                                        oa.store(recip_w * gv * dlv - scale_w * iv * recip_w * recip_w * inv_c_w)
                                    with out_dg_blk.reserve() as og:
                                        og.store(iv * recip_w * dlv)

        @ttl.datamovement()
        def dm_read():
            core_col, core_row = ttl.node(dims=2)
            core_linear = core_row * grid_cols + core_col
            for local_row in range(rows_per_core):
                r = core_linear * rows_per_core + local_row
                if r < ht:
                    # Pass 1
                    with rms_tile.reserve() as b:
                        ttl.copy(rms_t[r, 0], b).wait()
                    for blk_idx in range(num_col_blocks):
                        col_start = blk_idx * block_size
                        col_end = col_start + block_size
                        with (
                            inp_blk.reserve() as i_blk,
                            gamma_blk.reserve() as g_blk,
                            dL_blk.reserve() as d_blk,
                        ):
                            i_tx = ttl.copy(input_t[r, col_start:col_end], i_blk)
                            g_tx = ttl.copy(gamma_t[0, col_start:col_end], g_blk)
                            d_tx = ttl.copy(dL_dout_t[r, col_start:col_end], d_blk)
                            i_tx.wait()
                            g_tx.wait()
                            d_tx.wait()
                    # Pass 2
                    with rms_tile.reserve() as b:
                        ttl.copy(rms_t[r, 0], b).wait()
                    for blk_idx in range(num_col_blocks):
                        col_start = blk_idx * block_size
                        col_end = col_start + block_size
                        with (
                            inp_blk.reserve() as i_blk,
                            gamma_blk.reserve() as g_blk,
                            dL_blk.reserve() as d_blk,
                        ):
                            i_tx = ttl.copy(input_t[r, col_start:col_end], i_blk)
                            g_tx = ttl.copy(gamma_t[0, col_start:col_end], g_blk)
                            d_tx = ttl.copy(dL_dout_t[r, col_start:col_end], d_blk)
                            i_tx.wait()
                            g_tx.wait()
                            d_tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_col, core_row = ttl.node(dims=2)
            core_linear = core_row * grid_cols + core_col
            for local_row in range(rows_per_core):
                r = core_linear * rows_per_core + local_row
                if r < ht:
                    for blk_idx in range(num_col_blocks):
                        col_start = blk_idx * block_size
                        col_end = col_start + block_size
                        with out_da_blk.wait() as b:
                            ttl.copy(b, dL_dinput_out[r, col_start:col_end]).wait()
                        with out_dg_blk.wait() as b:
                            ttl.copy(b, dL_dgamma_comp_out[r, col_start:col_end]).wait()

    return rmsnorm_bw_2pass


def run_rmsnorm_bw_2pass(device, kernel, x_p, g_p, rms_p, dL_p, out_da, out_dg):
    """Run 2-pass TTL kernel; reduce per-row dL/dgamma components like ``ttml::rmsnorm_bw``."""
    kernel(x_p, g_p, rms_p, dL_p, out_da, out_dg)
    dg_reduced = ttnn.sum(out_dg, dim=[0], keepdim=True)
    return out_da, dg_reduced


def to_dev(t, device):
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def pad(t, rows_padded, cols_padded):
    rows, cols = t.shape
    pad_r = rows_padded - rows
    pad_c = cols_padded - cols
    if pad_r == 0 and pad_c == 0:
        return t
    return torch.nn.functional.pad(t, (0, pad_c, 0, pad_r), value=0.0)
