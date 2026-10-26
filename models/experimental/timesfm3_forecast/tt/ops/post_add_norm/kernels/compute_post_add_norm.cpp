// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TimesFM-3 post-norm + residual + next pre-norm, one tile row at a time:
//   h  = res + o * rsqrt(mean(o^2) + eps) * gain
//   xn = h * rsqrt(mean(h^2) + eps)
// The o row stays in its input buffer for both of its passes; o * rstd and its gain product go through fp32
// row buffers (t, t2), so the only bf16 roundings are the ones the 3-op chain has (h and xn). h is packed to the
// writer and to a row buffer the second norm re-reads.
//
// Compile-time args: Wt, blk (divides Wt), inv_w (fp32 bits), eps (fp32 bits). Runtime: num_rows.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t blk = get_compile_time_arg_val(1);
constexpr uint32_t inv_w = get_compile_time_arg_val(2);
constexpr uint32_t eps = get_compile_time_arg_val(3);

constexpr uint32_t o_cb = tt::CBIndex::c_0;
constexpr uint32_t r_cb = tt::CBIndex::c_1;
constexpr uint32_t scaler_cb = tt::CBIndex::c_2;
constexpr uint32_t g_cb = tt::CBIndex::c_3;
constexpr uint32_t h_out_cb = tt::CBIndex::c_16;
constexpr uint32_t n_out_cb = tt::CBIndex::c_17;
constexpr uint32_t h_cb = tt::CBIndex::c_24;
constexpr uint32_t sq_cb = tt::CBIndex::c_25;
constexpr uint32_t rstd_cb = tt::CBIndex::c_26;
constexpr uint32_t t_cb = tt::CBIndex::c_27;
constexpr uint32_t t2_cb = tt::CBIndex::c_28;

// rstd of a whole row held in ``src`` (waited for block by block, not popped) -> one tile in rstd_cb.
inline void row_rstd(uint32_t src) {
    CircularBuffer cb_src(src);
    CircularBuffer cb_sq(sq_cb);
    CircularBuffer cb_rstd(rstd_cb);
    reconfig_data_format(src, src);
    pack_reconfig_data_format(sq_cb);
    mul_init(src, src);
    for (uint32_t j0 = 0; j0 < Wt; j0 += blk) {
        cb_src.wait_front(j0 + blk);
        cb_sq.reserve_back(blk);
        tile_regs_acquire();
        for (uint32_t d = 0; d < blk; ++d) {
            mul_tiles(src, src, j0 + d, j0 + d, d);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t d = 0; d < blk; ++d) {
            pack_tile(d, sq_cb);
        }
        tile_regs_release();
        cb_sq.push_back(blk);
    }
    cb_sq.wait_front(Wt);
    reconfig_data_format(scaler_cb, sq_cb);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(sq_cb, scaler_cb, rstd_cb);
    tile_regs_acquire();
    for (uint32_t j = 0; j < Wt; ++j) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(sq_cb, scaler_cb, j, 0, 0);
    }
    reduce_uninit();
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_w);
    add_unary_tile(0, eps);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    cb_sq.pop_front(Wt);
    cb_rstd.reserve_back(1);
    pack_reconfig_data_format(rstd_cb);
    tile_regs_wait();
    pack_tile(0, rstd_cb);
    tile_regs_release();
    cb_rstd.push_back(1);
}

// dst row (Wt tiles, in blocks) = src row * rstd, broadcast over columns.
inline void scale_row(uint32_t src, uint32_t dst) {
    CircularBuffer cb_dst(dst);
    reconfig_data_format(src, rstd_cb);
    pack_reconfig_data_format(dst);
    mul_bcast_cols_init(src, rstd_cb);
    for (uint32_t j0 = 0; j0 < Wt; j0 += blk) {
        cb_dst.reserve_back(blk);
        tile_regs_acquire();
        for (uint32_t d = 0; d < blk; ++d) {
            mul_tiles_bcast_cols(src, rstd_cb, j0 + d, 0, d);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t d = 0; d < blk; ++d) {
            pack_tile(d, dst);
        }
        tile_regs_release();
        cb_dst.push_back(blk);
    }
}

void kernel_main() {
    const uint32_t num_rows = get_arg_val<uint32_t>(0);

    CircularBuffer cb_o(o_cb);
    CircularBuffer cb_r(r_cb);
    CircularBuffer cb_scaler(scaler_cb);
    CircularBuffer cb_g(g_cb);
    CircularBuffer cb_h_out(h_out_cb);
    CircularBuffer cb_h(h_cb);
    CircularBuffer cb_rstd(rstd_cb);
    CircularBuffer cb_t(t_cb);
    CircularBuffer cb_t2(t2_cb);

    compute_kernel_hw_startup(o_cb, o_cb, sq_cb);
    cb_scaler.wait_front(1);
    cb_g.wait_front(Wt);

    for (uint32_t row = 0; row < num_rows; ++row) {
        // t = o * rstd(o)
        row_rstd(o_cb);
        cb_rstd.wait_front(1);
        scale_row(o_cb, t_cb);
        cb_rstd.pop_front(1);
        cb_o.pop_front(Wt);

        // t2 = t * gain (row 0 of each gain tile, broadcast over rows)
        reconfig_data_format(t_cb, g_cb);
        pack_reconfig_data_format(t2_cb);
        mul_bcast_rows_init(t_cb, g_cb);
        for (uint32_t j0 = 0; j0 < Wt; j0 += blk) {
            cb_t.wait_front(blk);
            cb_t2.reserve_back(blk);
            tile_regs_acquire();
            for (uint32_t d = 0; d < blk; ++d) {
                mul_tiles_bcast_rows(t_cb, g_cb, d, j0 + d, d);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t d = 0; d < blk; ++d) {
                pack_tile(d, t2_cb);
            }
            tile_regs_release();
            cb_t.pop_front(blk);
            cb_t2.push_back(blk);
        }

        // h = res + t2, to the writer and to the row buffer
        reconfig_data_format(r_cb, t2_cb);
        pack_reconfig_data_format(h_cb);
        add_init(r_cb, t2_cb);
        for (uint32_t j0 = 0; j0 < Wt; j0 += blk) {
            cb_r.wait_front(blk);
            cb_t2.wait_front(blk);
            cb_h_out.reserve_back(blk);
            cb_h.reserve_back(blk);
            tile_regs_acquire();
            for (uint32_t d = 0; d < blk; ++d) {
                add_tiles(r_cb, t2_cb, d, d, d);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t d = 0; d < blk; ++d) {
                pack_tile(d, h_out_cb);
            }
            for (uint32_t d = 0; d < blk; ++d) {
                pack_tile(d, h_cb);
            }
            tile_regs_release();
            cb_r.pop_front(blk);
            cb_t2.pop_front(blk);
            cb_h_out.push_back(blk);
            cb_h.push_back(blk);
        }

        // xn = h * rstd(h)
        row_rstd(h_cb);
        cb_rstd.wait_front(1);
        scale_row(h_cb, n_out_cb);
        cb_rstd.pop_front(1);
        cb_h.pop_front(Wt);
    }
}
