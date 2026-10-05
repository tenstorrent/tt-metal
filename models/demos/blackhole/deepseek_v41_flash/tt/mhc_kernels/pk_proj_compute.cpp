// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// P_{r,k} = sum_j X_{r,j} @ W_j + sum_j (X_{r,j} (.) X_{r,j}) @ ONES_col  -- partial projection (column NCOL: partial
// sum of squares) per token tile row r.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(2);
    constexpr uint32_t cb_sq = get_compile_time_arg_val(3);
    constexpr uint32_t cb_p = get_compile_time_arg_val(4);
    constexpr uint32_t TPC = get_compile_time_arg_val(5);
    constexpr uint32_t R = get_compile_time_arg_val(6);

    compute_kernel_hw_startup(cb_x, cb_w, cb_p);
    CircularBuffer xx(cb_x), w(cb_w), ones(cb_ones), sq(cb_sq), p(cb_p);
    w.wait_front(TPC);
    ones.wait_front(1);
    for (uint32_t r = 0; r < R; ++r) {
        xx.wait_front(TPC);
        reconfig_data_format(cb_x, cb_x);
        mul_tiles_init(cb_x, cb_x);
        for (uint32_t j = 0; j < TPC; ++j) {
            tile_regs_acquire();
            mul_tiles(cb_x, cb_x, j, j, 0);
            tile_regs_commit();
            tile_regs_wait();
            sq.reserve_back(1);
            pack_tile(0, cb_sq);
            sq.push_back(1);
            tile_regs_release();
        }
        sq.wait_front(TPC);
        tile_regs_acquire();
        reconfig_data_format(cb_x, cb_w);
        matmul_init(cb_x, cb_w);
        for (uint32_t j = 0; j < TPC; ++j) {
            matmul_tiles(cb_x, cb_w, j, j, 0);
        }
        reconfig_data_format(cb_sq, cb_ones);
        matmul_init(cb_sq, cb_ones);
        for (uint32_t j = 0; j < TPC; ++j) {
            matmul_tiles(cb_sq, cb_ones, j, 0, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        p.reserve_back(1);
        pack_tile(0, cb_p);
        p.push_back(1);
        tile_regs_release();
        sq.pop_front(TPC);
        xx.pop_front(TPC);
    }
}
