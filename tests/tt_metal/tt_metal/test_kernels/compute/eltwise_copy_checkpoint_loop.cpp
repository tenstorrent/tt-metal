// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel: exercises checkpoint in a loop and DEBUG_CHECKPOINT_EX with dump_dest.
// Verifies that the epoch counter handles repeated barriers correctly.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/debug/dprint.h"
#include "api/debug/checkpoint.h"

void kernel_main() {
    uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    unary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_16);

    acquire_dst();
    cb_wait_front(tt::CBIndex::c_0, per_core_tile_cnt);
    cb_reserve_back(tt::CBIndex::c_16, per_core_tile_cnt);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        copy_tile(tt::CBIndex::c_0, b, b);
    }

    // Test 1: checkpoint in a loop (same ID reused — epoch counter must handle this)
    for (uint32_t i = 0; i < 3; i++) {
        DEBUG_CHECKPOINT(1);
    }

    // Test 2: DEBUG_CHECKPOINT_EX with dump_dest=true (Math thread dumps dest regs)
    DEBUG_CHECKPOINT_EX(2, 2, 0, true);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        pack_tile(b, tt::CBIndex::c_16);
        cb_pop_front(tt::CBIndex::c_0, 1);
        cb_push_back(tt::CBIndex::c_16, 1);
    }

    release_dst();
}
