# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn.functional as F
import ttnn
import ttl

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
    bcw = 1

    @ttl.operation(grid="auto", fp32_dest_acc_en=True)
    def rmsnorm_bw_device_kernels_ttl(
        input_t,
        gamma_t,
        rms_t,
        dL_dout_t,
        dL_dinput_out,
        dL_dgamma_comp_out,
    ):
        ht = input_t.shape[0] // TILE_WIDTH
        wt = input_t.shape[1] // TILE_WIDTH
        block_size = get_block_size(wt)
        bs = 2
        # DFB tile shape for one reader/writer "push": one tile-row × block_size column tiles
        tr, tc = 1, bs
        # Inner width in elements for one full tile-row strip
        elem_c = wt * TILE_WIDTH
        # 1/C for RMS mean denominator;
        inv_c_scalar = 1.0 / elem_c
        # Row-partition across cores: rows reduce along C independently, so each core owns whole rows.
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_cores = grid_cols * grid_rows
        rows_per_core = -(-ht // total_cores)  # divceil
        col_blocks = wt // bs

        inp_dfb = ttl.make_dataflow_buffer_like(input_t, shape=(tr, tc), block_count=bc)
        gamma_dfb = ttl.make_dataflow_buffer_like(gamma_t, shape=(tr, tc), block_count=bc)
        rms_dfb = ttl.make_dataflow_buffer_like(rms_t, shape=(tr, tc), block_count=bc)
        dL_dfb = ttl.make_dataflow_buffer_like(dL_dout_t, shape=(tr, tc), block_count=bc)

        # Full tile-row strips for scale = sum_c (a * gained)
        wide_row_shape = (1, wt)
        wide_inp = ttl.make_dataflow_buffer_like(input_t, shape=wide_row_shape, block_count=bcw)
        wide_gam = ttl.make_dataflow_buffer_like(gamma_t, shape=wide_row_shape, block_count=bcw)
        wide_rms = ttl.make_dataflow_buffer_like(rms_t, shape=wide_row_shape, block_count=bcw)
        wide_dL = ttl.make_dataflow_buffer_like(dL_dout_t, shape=wide_row_shape, block_count=bcw)

        # --- Scale path (compute_scale): reduce_sum wants a scaler tile; contrib must be materialized first. ---
        # Unity scaler for ttl.math.reduce_sum
        single_tile_shape = (1, 1)
        scaler_dfb = ttl.make_dataflow_buffer_like(input_t, shape=single_tile_shape, block_count=bc)
        # a * gained over the full (1, Wt) strip
        contrib_wide_dfb = ttl.make_dataflow_buffer_like(input_t, shape=wide_row_shape, block_count=bcw)
        # Reduced scale along horizontal tiles (dim C)
        scale_red_dfb = ttl.make_dataflow_buffer_like(input_t, shape=single_tile_shape, block_count=bc)
        # Broadcast scale to current block for mul_bcast_cols-style use in dL_da rhs
        scale_bc_blk_dfb = ttl.make_dataflow_buffer_like(input_t, shape=(tr, tc), block_count=bc)
        # Constant 1/C tile for rhs
        inv_c_dfb = ttl.make_dataflow_buffer_like(input_t, shape=(tr, tc), block_count=bc)

        # --- dL_da / dL_dgamma intermediates  ---
        # gained_dfb = ttl.make_dataflow_buffer_like(input_t, shape=(tr, tc), block_count=bc)
        out_da_dfb = ttl.make_dataflow_buffer_like(dL_dinput_out, shape=(tr, tc), block_count=bc)
        out_dg_dfb = ttl.make_dataflow_buffer_like(dL_dgamma_comp_out, shape=(tr, tc), block_count=bc)

        @ttl.compute()
        def compute():
            core_col, core_row = ttl.node(dims=2)
            core_linear = core_row * grid_cols + core_col
            with inv_c_dfb.reserve() as ic:
                ic.store(ttl.math.fill(ic, inv_c_scalar))
            icv = inv_c_dfb.wait()
            with scaler_dfb.reserve() as sc:
                sc.store(ttl.math.fill(sc, 1.0))
            scv = scaler_dfb.wait()
            for local_row in range(rows_per_core):
                row = core_linear * rows_per_core + local_row
                if row < ht:
                    with (
                        wide_inp.wait() as wix,
                        wide_gam.wait() as wgam,
                        wide_rms.wait() as wrms,
                        wide_dL.wait() as wdl,
                    ):
                        recip_w = ttl.math.recip(wrms)
                        gained_w = recip_w * wgam * wdl
                        with contrib_wide_dfb.reserve() as cw:
                            cw.store(wix * gained_w)

                        # Computes scale factor: scale = sum(a * gained_dL_dout, dim=C)
                        # where gained_dL_dout = (gamma / rms_a) * dL_dout.
                        with scale_red_dfb.reserve() as sr:
                            sr.store(ttl.math.reduce_sum(contrib_wide_dfb.wait(), scv, dims=[1]))
                        srt = scale_red_dfb.wait()
                        with scale_red_dfb.wait() as srt, scale_bc_blk_dfb.reserve() as sbb:
                            sbb.store(ttl.math.broadcast(srt, sbb, dims=[1]))
                        sbv = scale_bc_blk_dfb.wait()
                        for _ in range(col_blocks):
                            with (
                                inp_dfb.wait() as iv,
                                gamma_dfb.wait() as gv,
                                rms_dfb.wait() as rmv,
                                dL_dfb.wait() as dlv,
                            ):
                                # Per-tile 1/rms for this block
                                recip_b = ttl.math.recip(rmv)
                                with out_da_dfb.reserve() as oa:
                                    oa.store(recip_b * gv * dlv - sbv * iv * recip_b * recip_b * icv)
                                with out_dg_dfb.reserve() as og:
                                    og.store(iv * recip_b * dlv)

        @ttl.datamovement()
        def dm_read():
            core_col, core_row = ttl.node(dims=2)
            core_linear = core_row * grid_cols + core_col
            for local_row in range(rows_per_core):
                r = core_linear * rows_per_core + local_row
                if r < ht:
                    with (
                        wide_inp.reserve() as wi_blk,
                        wide_gam.reserve() as wg_blk,
                        wide_rms.reserve() as wr_blk,
                        wide_dL.reserve() as wd_blk,
                    ):
                        wi_tx = ttl.copy(input_t[r : r + 1, 0:wt], wi_blk)
                        wg_tx = ttl.copy(gamma_t[r : r + 1, 0:wt], wg_blk)
                        wr_tx = ttl.copy(rms_t[r : r + 1, 0:wt], wr_blk)
                        wd_tx = ttl.copy(dL_dout_t[r : r + 1, 0:wt], wd_blk)
                        wi_tx.wait()
                        wg_tx.wait()
                        wr_tx.wait()
                        wd_tx.wait()
                    for c in range(0, wt, bs):
                        with (
                            inp_dfb.reserve() as i_blk,
                            gamma_dfb.reserve() as g_blk,
                            rms_dfb.reserve() as r_blk,
                            dL_dfb.reserve() as d_blk,
                        ):
                            i_tx = ttl.copy(input_t[r : r + 1, c : c + bs], i_blk)
                            g_tx = ttl.copy(gamma_t[r : r + 1, c : c + bs], g_blk)
                            r_tx = ttl.copy(rms_t[r : r + 1, c : c + bs], r_blk)
                            d_tx = ttl.copy(dL_dout_t[r : r + 1, c : c + bs], d_blk)
                            i_tx.wait()
                            g_tx.wait()
                            r_tx.wait()
                            d_tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_col, core_row = ttl.node(dims=2)
            core_linear = core_row * grid_cols + core_col
            for local_row in range(rows_per_core):
                r = core_linear * rows_per_core + local_row
                if r < ht:
                    for c in range(0, wt, bs):
                        with out_da_dfb.wait() as blk:
                            tx = ttl.copy(blk, dL_dinput_out[r : r + 1, c : c + bs])
                            tx.wait()
                        with out_dg_dfb.wait() as blk:
                            tx = ttl.copy(blk, dL_dgamma_comp_out[r : r + 1, c : c + bs])
                            tx.wait()

    return rmsnorm_bw_device_kernels_ttl


