// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batch-1 LM head + greedy, compute (fused_decode/lmhead.py: LmHeadGreedyB1): per streamed vocab column, one custom_mm
// over K = kt (1x32 activation tiles, LoFi, FP32 DST) -> one 1x32 BF16 logits tile.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/pack.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t kt = get_compile_time_arg_val(3);
    const uint32_t cols = get_arg_val<uint32_t>(0);
    constexpr auto dest_layout = CustomMmDestLayout::DefaultSplitAcc;
    custom_mm_block_init<false, dest_layout>(cb_x, cb_w, cb_out);
    cb_wait_front(cb_x, kt);
    for (uint32_t j = 0; j < cols; ++j) {
        cb_wait_front(cb_w, kt);
        tile_regs_acquire();
        custom_mm_block<dest_layout>(cb_x, cb_w, 0, 0, 0, kt);
        cb_pop_front(cb_w, kt);
        tile_regs_commit();
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
    cb_pop_front(cb_x, kt);
    custom_mm_block_uninit<dest_layout>(cb_out);
}
