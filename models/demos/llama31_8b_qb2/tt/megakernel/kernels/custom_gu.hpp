// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "api/compute/experimental/custom_mm.h"

#ifdef TRISC_UNPACK
// Use the pinned source revision's custom MOP with an explicit weight row
// stride. Each bank's native layout is[K,28]; this call computes seven columns
// without a persistent weight permutation or extra weight payload.
void custom_gu_unpack(uint32_t column) {
    auto& weights = get_local_cb_interface(1);
    auto& input = get_local_cb_interface(0);
    volatile uint32_t* cfg = get_cfg_pointer();
    wait_for_next_context(1);
    reset_config_context();
    TTI_UNPACR_NOP(SrcB, 0, 0, 0, 0, 0, 1, 0, p_unpacr_nop::CLR_SRC);
    _llk_unpack_AB_custom_mm_run_(cfg,
        weights.fifo_rd_ptr - 1 + column * weights.fifo_page_size,
        input.fifo_rd_ptr - 1, weights.fifo_page_size, 22 * weights.fifo_page_size, 8);
}
#endif

void custom_gu_projection() {
    // A is8 rows; partial/output stay16 rows to retain face1 atDSTrow16.
    // Original K-block8, BF16 L1 partials, LoFi, no split accumulation.
    custom_mm_block_init_short<false, false, false>(0, 1, 24, 7);
    pack_reconfig_data_format(24);
    pack_reconfig_l1_acc(0);
    for (uint32_t block = 0; block < 16; ++block) {
        cb_wait_front(0, 8);
        cb_wait_front(1, 224);
        const bool last = block == 15;
        for (uint32_t column = 0; column < 28; column += 7) {
            tile_regs_acquire();
            if (last) {
                reconfig_data_format_srca(1, 24);
                copy_init(24);
                cb_wait_front(24, 7);
                copy_block(24, 0, 0, 7);
                cb_pop_front(24, 7);
                reconfig_data_format_srca(24, 1);
                custom_mm_block_init_short<false, false, false>(0, 1, 24, 7);
            }
            UNPACK((custom_gu_unpack(column)));
            MATH((llk_math_custom_mm<false>(0, 1, 0, 8, 7)));
            tile_regs_commit();
            const uint32_t destination = last ? 16 : 24;
            cb_reserve_back(destination, 7);
            tile_regs_wait();
            pack_reconfig_data_format(destination);
            pack_reconfig_l1_acc(!last && block > 0);
            pack_block(0, destination, 7);
            tile_regs_release();
            cb_push_back(destination, 7);
        }
        if (block < 14) {
            for (uint32_t column = 0; column < 28; column += 7) {
                cb_wait_front(24, 7);
                cb_pop_front(24, 7);
            }
        }
        cb_pop_front(0, 8);
        cb_pop_front(1, 224);
    }
    pack_reconfig_l1_acc(0);
    custom_mm_block_uninit<false>();
}
