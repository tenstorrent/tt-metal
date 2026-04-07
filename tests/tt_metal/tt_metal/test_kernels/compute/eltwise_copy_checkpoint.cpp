// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel for debug checkpoints in fused kernels.
// Copies tiles from input CB to output CB with a checkpoint between two stages,
// with a checkpoint between stages for fine-grain debugging of a large op.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/debug/dprint.h"
#include "api/debug/checkpoint.h"

void kernel_main() {
    uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    unary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_16);

    // Stage 1: Copy tiles from input CB to dest registers
    acquire_dst();
    cb_wait_front(tt::CBIndex::c_0, per_core_tile_cnt);
    cb_reserve_back(tt::CBIndex::c_16, per_core_tile_cnt);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        copy_tile(tt::CBIndex::c_0, b, b);
    }

    // Checkpoint between stages: all RISCs synchronize and dump CB state
    DEBUG_CHECKPOINT(1);

    // Stage 2: Pack tiles from dest registers to output CB
    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        pack_tile(b, tt::CBIndex::c_16);
        cb_pop_front(tt::CBIndex::c_0, 1);
        cb_push_back(tt::CBIndex::c_16, 1);
    }

    release_dst();
}
