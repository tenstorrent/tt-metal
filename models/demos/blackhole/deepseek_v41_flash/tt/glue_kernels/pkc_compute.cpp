// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// T19 packed-mHC collapse (+ sum of squares), compute.  Collapse as pk_comb_compute.cpp (NOUT == 1): out tile = sum_i
// diag(pre_i) @ x_i (fp32 DST) packed to bf16 into cb_out (-> writer) AND cb_h2 (kept); then the squares of the bf16
// values of this core's cs tiles are summed per token row through a matmul with a ones column (as pk_proj does) into
// ONE partial tile (column 0 = partial sum of squares of the 32 tokens).
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_i = get_compile_time_arg_val(0);
    constexpr uint32_t cb_ct = get_compile_time_arg_val(1);
    constexpr uint32_t cb_x = get_compile_time_arg_val(2);
    constexpr uint32_t cb_d = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);
    constexpr uint32_t cb_h2 = get_compile_time_arg_val(5);
    constexpr uint32_t cb_sq = get_compile_time_arg_val(6);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(7);
    constexpr uint32_t cb_p = get_compile_time_arg_val(8);
    constexpr uint32_t NC = get_compile_time_arg_val(9);  // = 4 pre tiles
    constexpr uint32_t BLK = get_compile_time_arg_val(10);
    const uint32_t ncols = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_x, cb_d, cb_out);
    CircularBuffer ci_(cb_i), ct(cb_ct), x(cb_x), d(cb_d), out(cb_out), h2(cb_h2), sq(cb_sq), ones(cb_ones), p(cb_p);

    ci_.wait_front(1);
    ct.wait_front(NC);
    reconfig_data_format(cb_i, cb_ct);
    pack_reconfig_data_format(cb_d);
    mul_bcast_cols_init(cb_i, cb_ct);
    for (uint32_t q = 0; q < NC; ++q) {
        tile_regs_acquire();
        mul_tiles_bcast_cols(cb_i, cb_ct, 0, q, 0);
        tile_regs_commit();
        tile_regs_wait();
        d.reserve_back(1);
        pack_tile(0, cb_d);
        d.push_back(1);
        tile_regs_release();
    }
    d.wait_front(NC);
    pack_reconfig_data_format(cb_out);

    for (uint32_t b = 0; b < ncols; b += BLK) {
        x.wait_front(4 * BLK);
        for (uint32_t ci = 0; ci < BLK; ++ci) {
            tile_regs_acquire();
            reconfig_data_format(cb_d, cb_x);
            matmul_init(cb_d, cb_x);
            for (uint32_t i = 0; i < 4; ++i) {
                matmul_tiles(cb_d, cb_x, i, ci * 4 + i, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            out.reserve_back(1);
            h2.reserve_back(1);
            pack_tile(0, cb_out);
            pack_tile(0, cb_h2);
            out.push_back(1);
            h2.push_back(1);
            tile_regs_release();
        }
        x.pop_front(4 * BLK);
    }

    // partial sum of squares of the bf16 outputs of this core
    ones.wait_front(1);
    h2.wait_front(ncols);
    reconfig_data_format(cb_h2, cb_h2);
    pack_reconfig_data_format(cb_sq);
    mul_init(cb_h2, cb_h2);
    for (uint32_t j = 0; j < ncols; ++j) {
        tile_regs_acquire();
        mul_tiles(cb_h2, cb_h2, j, j, 0);
        tile_regs_commit();
        tile_regs_wait();
        sq.reserve_back(1);
        pack_tile(0, cb_sq);
        sq.push_back(1);
        tile_regs_release();
    }
    sq.wait_front(ncols);
    tile_regs_acquire();
    reconfig_data_format(cb_sq, cb_ones);
    matmul_init(cb_sq, cb_ones);
    for (uint32_t j = 0; j < ncols; ++j) {
        matmul_tiles(cb_sq, cb_ones, j, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_p);
    p.reserve_back(1);
    pack_tile(0, cb_p);
    p.push_back(1);
    tile_regs_release();
    sq.pop_front(ncols);
    h2.pop_front(ncols);
}
