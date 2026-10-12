// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// [y = a + b,] n = y * rsqrt(mean(y^2) + eps), one tile row at a time in blocks of blk tiles. When fused,
// y is packed twice: to the writer's buffer and to a row buffer this kernel re-reads for the square and
// the final scale; otherwise the input row buffer is kept until the row is done.
//
// Compile-time args: Wt, blk (tiles per DST pass, divides Wt), inv_w (fp32 bits), eps (fp32 bits), fuse_add
// Runtime args: num_rows

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);
    constexpr uint32_t inv_w = get_compile_time_arg_val(2);
    constexpr uint32_t eps = get_compile_time_arg_val(3);
    constexpr bool fuse_add = get_compile_time_arg_val(4) != 0;
    const uint32_t num_rows = get_arg_val<uint32_t>(0);

    constexpr uint32_t a_cb = tt::CBIndex::c_0;
    constexpr uint32_t b_cb = tt::CBIndex::c_1;
    constexpr uint32_t scaler_cb = tt::CBIndex::c_2;
    constexpr uint32_t y_out_cb = tt::CBIndex::c_16;
    constexpr uint32_t n_out_cb = tt::CBIndex::c_17;
    constexpr uint32_t y_cb = fuse_add ? tt::CBIndex::c_24 : a_cb;
    constexpr uint32_t sq_cb = tt::CBIndex::c_25;
    constexpr uint32_t rstd_cb = tt::CBIndex::c_26;

    CircularBuffer cb_a(a_cb);
    CircularBuffer cb_b(b_cb);
    CircularBuffer cb_scaler(scaler_cb);
    CircularBuffer cb_y_out(y_out_cb);
    CircularBuffer cb_n_out(n_out_cb);
    CircularBuffer cb_y(y_cb);
    CircularBuffer cb_sq(sq_cb);
    CircularBuffer cb_rstd(rstd_cb);

    if constexpr (fuse_add) {
        compute_kernel_hw_startup(a_cb, b_cb, y_cb);
    } else {
        compute_kernel_hw_startup(a_cb, a_cb, sq_cb);
    }
    cb_scaler.wait_front(1);

    for (uint32_t row = 0; row < num_rows; ++row) {
        if constexpr (fuse_add) {
            reconfig_data_format(a_cb, b_cb);
            pack_reconfig_data_format(y_cb);
            add_init(a_cb, b_cb);
            for (uint32_t j0 = 0; j0 < Wt; j0 += blk) {
                cb_a.wait_front(blk);
                cb_b.wait_front(blk);
                cb_y_out.reserve_back(blk);
                cb_y.reserve_back(blk);
                tile_regs_acquire();
                for (uint32_t d = 0; d < blk; ++d) {
                    add_tiles(a_cb, b_cb, d, d, d);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t d = 0; d < blk; ++d) {
                    pack_tile(d, y_out_cb);
                }
                for (uint32_t d = 0; d < blk; ++d) {
                    pack_tile(d, y_cb);
                }
                tile_regs_release();
                cb_a.pop_front(blk);
                cb_b.pop_front(blk);
                cb_y_out.push_back(blk);
                cb_y.push_back(blk);
            }
        }

        // y^2, starting as soon as the first block of y is in.
        reconfig_data_format(y_cb, y_cb);
        pack_reconfig_data_format(sq_cb);
        mul_init(y_cb, y_cb);
        for (uint32_t j0 = 0; j0 < Wt; j0 += blk) {
            cb_y.wait_front(j0 + blk);
            cb_sq.reserve_back(blk);
            tile_regs_acquire();
            for (uint32_t d = 0; d < blk; ++d) {
                mul_tiles(y_cb, y_cb, j0 + d, j0 + d, d);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t d = 0; d < blk; ++d) {
                pack_tile(d, sq_cb);
            }
            tile_regs_release();
            cb_sq.push_back(blk);
        }

        // rstd = rsqrt(sum(y^2) / W + eps), in column 0.
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

        // n = y * rstd
        cb_rstd.wait_front(1);
        reconfig_data_format(y_cb, rstd_cb);
        pack_reconfig_data_format(n_out_cb);
        mul_bcast_cols_init(y_cb, rstd_cb);
        for (uint32_t j0 = 0; j0 < Wt; j0 += blk) {
            cb_n_out.reserve_back(blk);
            tile_regs_acquire();
            for (uint32_t d = 0; d < blk; ++d) {
                mul_tiles_bcast_cols(y_cb, rstd_cb, j0 + d, 0, d);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t d = 0; d < blk; ++d) {
                pack_tile(d, n_out_cb);
            }
            tile_regs_release();
            cb_n_out.push_back(blk);
        }
        cb_rstd.pop_front(1);
        cb_y.pop_front(Wt);
    }
}
