// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Streamed matmul (TRISC) for the decode stream ops (experts/stream.py): one 1x32 output tile per weight column,
// out[col] = in0 . W[:, col] over kt K tiles (1x32 activation tiles, custom_mm, LoFi, FP32 accumulation). The down
// stream uses it for the score-weighted expert sum (K = k segments [w_e * act_e | w_e]), LinearStream for a dense
// projection + bias.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/pack.h"

void kernel_main() {
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t kt = get_compile_time_arg_val(3);
    constexpr uint32_t cols = get_compile_time_arg_val(4);

    constexpr bool transpose = false;
    constexpr bool split_acc = true;
    constexpr bool dense_packing = false;
    custom_mm_block_init<transpose, split_acc, dense_packing>(cb_in0, cb_w, cb_out);

    cb_wait_front(cb_in0, kt);
    for (uint32_t c = 0; c < cols; ++c) {
        tile_regs_acquire();
        cb_wait_front(cb_w, kt);
        custom_mm_block<true>(cb_in0, cb_w, 0, 0, 0, kt);
        cb_pop_front(cb_w, kt);
        tile_regs_commit();
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
    cb_pop_front(cb_in0, kt);
    custom_mm_block_uninit<dense_packing>();
}
