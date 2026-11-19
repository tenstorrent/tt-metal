// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Row exchange, worker compute (TRISC): untilizes the local chunk (CT tiles -> the row-major [32, CT * 32] block the
// writer cuts the pieces from), tilizes the assembled block the writer builds from every row's pieces, and for the
// reduce multiplies every assembled tile by the selector S (fp32 DST, HiFi4: out[t, :] = sum_r block[r * T + t, :],
// one bf16 x 1.0 product per term, rounded once).
//
// MODE 2 (all-reduce over the columns): the owner core of a chunk sums the sources like the reduce and untilizes the
// sums (packed a second time into CB_OUT2) into CB_RM3 for the phase-2 pieces; a non-owner only tilizes.
//
// compile-time: [MODE, CB_TIN, CB_RM, CB_RM2, CB_T, CB_SEL, CB_OUT, CT, OWNER, CB_OUT2, CB_RM3]
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t MODE = get_compile_time_arg_val(0);
    constexpr uint32_t CB_TIN = get_compile_time_arg_val(1);
    constexpr uint32_t CB_RM = get_compile_time_arg_val(2);
    constexpr uint32_t CB_RM2 = get_compile_time_arg_val(3);
    constexpr uint32_t CB_T = get_compile_time_arg_val(4);
    constexpr uint32_t CB_SEL = get_compile_time_arg_val(5);
    constexpr uint32_t CB_OUT = get_compile_time_arg_val(6);
    constexpr uint32_t CT = get_compile_time_arg_val(7);
    constexpr uint32_t OWNER = get_compile_time_arg_val(8);
    constexpr uint32_t CB_OUT2 = get_compile_time_arg_val(9);
    constexpr uint32_t CB_RM3 = get_compile_time_arg_val(10);
    constexpr bool SUM = (MODE == 1) || (MODE == 2 && OWNER == 1);

    compute_kernel_hw_startup(CB_TIN, CB_RM);

    // 1. the local chunk -> row-major rows (the helper waits CB_TIN, reserves / pushes CB_RM, pops CB_TIN)
    compute_kernel_lib::untilize<CT, CB_TIN, CB_RM>(1);

    // 2. the assembled rows -> tiles
    cb_wait_front(CB_RM2, CT);
    cb_reserve_back(CB_T, CT);
    tilize_init(CB_RM2, CT, CB_T);
    tilize_block(CB_RM2, CT, CB_T);
    tilize_uninit(CB_RM2, CB_T);
    cb_push_back(CB_T, CT);
    cb_pop_front(CB_RM2, CT);

    if constexpr (SUM) {
        // 3. the row sum as one exact selector matmul per tile
        cb_wait_front(CB_T, CT);
        cb_wait_front(CB_SEL, 1);
        matmul_init(CB_SEL, CB_T);
        for (uint32_t j = 0; j < CT; ++j) {
            tile_regs_acquire();
            matmul_tiles(CB_SEL, CB_T, 0, j, 0);
            tile_regs_commit();
            cb_reserve_back(CB_OUT, 1);
            tile_regs_wait();
            pack_tile(0, CB_OUT);
            if constexpr (MODE == 2) {
                cb_reserve_back(CB_OUT2, 1);
                pack_tile(0, CB_OUT2);
                cb_push_back(CB_OUT2, 1);
            }
            tile_regs_release();
            cb_push_back(CB_OUT, 1);
        }
        cb_pop_front(CB_SEL, 1);
        cb_pop_front(CB_T, CT);
        if constexpr (MODE == 2) {
            // 4. the summed rows back to row-major for the phase-2 pieces
            compute_kernel_lib::untilize<CT, CB_OUT2, CB_RM3>(1);
        }
    }
}
