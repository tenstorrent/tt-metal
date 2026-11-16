// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// moe_decode down compute (TRISC): out[col] = sum_s h_s . Wd_s[:, col] as one custom_mm over K = k * kts tiles
// (1x32 activation tiles, LoFi, fp32 dest), one 1x32 output tile.
// compile-time args: [cb_in0, cb_w, cb_out, kt, kts]
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/pack.h"

void kernel_main() {
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t kt = get_compile_time_arg_val(3);
    constexpr bool transpose = false;
    constexpr auto dest_layout = CustomMmDestLayout::DefaultSplitAcc;
    custom_mm_block_init<transpose, dest_layout>(cb_in0, cb_w, cb_out);
    cb_wait_front(cb_in0, kt);
    cb_wait_front(cb_w, kt);
#ifdef SPLIT_SLOTS
    // two outputs: slots [0, SPLIT_SLOTS) and [SPLIT_SLOTS, k) (routed experts / shared-MLP virtual experts)
    constexpr uint32_t kts = get_compile_time_arg_val(4);
    constexpr uint32_t k0 = SPLIT_SLOTS * kts;
    cb_reserve_back(cb_out, 2);
    for (uint32_t part = 0; part < 2; ++part) {
        tile_regs_acquire();
        custom_mm_block<dest_layout>(cb_in0, cb_w, part ? k0 : 0, part ? k0 : 0, 0, part ? kt - k0 : k0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
    }
    cb_push_back(cb_out, 2);
    cb_pop_front(cb_w, kt);
#else
    tile_regs_acquire();
    custom_mm_block<dest_layout>(cb_in0, cb_w, 0, 0, 0, kt);
    cb_pop_front(cb_w, kt);
    tile_regs_commit();
    cb_reserve_back(cb_out, 1);
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
#endif
    cb_pop_front(cb_in0, kt);
    custom_mm_block_uninit<dest_layout>(cb_out);
}
