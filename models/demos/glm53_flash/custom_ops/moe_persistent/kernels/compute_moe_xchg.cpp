// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Persistent MoE program, exchange worker compute (TRISC): phase A untilizes the local gather chunk and tilizes
// the assembled block (custom_ops/row_exchange compute, MODE 0); phase B (workers with a reduce chunk) the same
// for the expert output chunk plus the exact selector sum (MODE 1).
//
// compile-time: [CB_TIN, CB_RM, CB_RM2, CB_T, CB_SEL, CB_OUT, CT_A, CT_B, HAS_B]
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace {
constexpr uint32_t CB_TIN = get_compile_time_arg_val(0);
constexpr uint32_t CB_RM = get_compile_time_arg_val(1);
constexpr uint32_t CB_RM2 = get_compile_time_arg_val(2);
constexpr uint32_t CB_T = get_compile_time_arg_val(3);
constexpr uint32_t CB_SEL = get_compile_time_arg_val(4);
constexpr uint32_t CB_OUT = get_compile_time_arg_val(5);
constexpr uint32_t CT_A = get_compile_time_arg_val(6);
constexpr uint32_t CT_B = get_compile_time_arg_val(7);
constexpr uint32_t HAS_B = get_compile_time_arg_val(8);

template <uint32_t CT>
FORCE_INLINE void untilize_then_tilize() {
    compute_kernel_lib::untilize<CT, CB_TIN, CB_RM>(1);
    cb_wait_front(CB_RM2, CT);
    cb_reserve_back(CB_T, CT);
    tilize_init(CB_RM2, CT, CB_T);
    tilize_block(CB_RM2, CT, CB_T);
    tilize_uninit(CB_RM2, CB_T);
    cb_push_back(CB_T, CT);
    cb_pop_front(CB_RM2, CT);
}
}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(CB_TIN, CB_RM);
    untilize_then_tilize<CT_A>();
    if constexpr (HAS_B) {
        untilize_then_tilize<CT_B>();
        cb_wait_front(CB_T, CT_B);
        cb_wait_front(CB_SEL, 1);
        matmul_init(CB_SEL, CB_T);
        for (uint32_t j = 0; j < CT_B; ++j) {
            tile_regs_acquire();
            matmul_tiles(CB_SEL, CB_T, 0, j, 0);
            tile_regs_commit();
            cb_reserve_back(CB_OUT, 1);
            tile_regs_wait();
            pack_tile(0, CB_OUT);
            tile_regs_release();
            cb_push_back(CB_OUT, 1);
        }
        cb_pop_front(CB_SEL, 1);
        cb_pop_front(CB_T, CT_B);
    }
}
