// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "api/compute/experimental/custom_mm.h"

void custom_qkv_projection() {
    // Original QKV K-block16, six output tiles, LoFi, BF16 L1 partials.
    // A uses8 rows; partial/output use16 rows so copy_block preserves the
    // custom math second-face destination atrow16. No split accumulation. The specialized MVMUL can change numerical
    // association, so qualification is explicit rather than assumed.
    custom_mm_block_init_short<false, false, false>(0, 1, 24, 6);
    pack_reconfig_data_format(24);
    pack_reconfig_l1_acc(0);
    for (uint32_t block = 0; block < 8; ++block) {
        cb_wait_front(0, 16);
        cb_wait_front(1, 96);
        const bool last = block == 7;
        tile_regs_acquire();
        if (last) {
            reconfig_data_format_srca(1, 24);
            copy_init(24);
            cb_wait_front(24, 6);
            copy_block(24, 0, 0, 6);
            cb_pop_front(24, 6);
            reconfig_data_format_srca(24, 1);
            custom_mm_block_init_short<false, false, false>(0, 1, 24, 6);
        }
        custom_mm_block<false>(0, 1, 0, 0, 0, 16, 6);
        tile_regs_commit();
        const uint32_t destination = last ? 16 : 24;
        cb_reserve_back(destination, 6);
        tile_regs_wait();
        pack_reconfig_data_format(destination);
        pack_reconfig_l1_acc(!last && block > 0);
        pack_block(0, destination, 6);
        tile_regs_release();
        cb_push_back(destination, 6);
        if (block < 6) {
            cb_wait_front(24, 6);
            cb_pop_front(24, 6);
        }
        cb_pop_front(0, 16);
        cb_pop_front(1, 96);
    }
    pack_reconfig_l1_acc(0);
    custom_mm_block_uninit<false>();
}
