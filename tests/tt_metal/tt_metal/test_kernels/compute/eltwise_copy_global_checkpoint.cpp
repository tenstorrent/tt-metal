// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel with global (cross-core) checkpoint support.
// Copies tiles with a global checkpoint at the micro-op boundary.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/debug/checkpoint.h"

void kernel_main() {
    uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);
    // Global checkpoint args (passed as compile-time args for compute).
    // TRISC doesn't execute the cross-core barrier, but still participates
    // in intra-core barriers.
    uint32_t sem_id = get_compile_time_arg_val(1);
    uint32_t barrier_coord_x = get_compile_time_arg_val(2);
    uint32_t barrier_coord_y = get_compile_time_arg_val(3);
    uint32_t num_cores = get_compile_time_arg_val(4);

    unary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_16);

    acquire_dst();
    cb_wait_front(tt::CBIndex::c_0, per_core_tile_cnt);
    cb_reserve_back(tt::CBIndex::c_16, per_core_tile_cnt);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        copy_tile(tt::CBIndex::c_0, b, b);
    }

    // Global checkpoint: synchronize all RISCs on all cores
    DEBUG_CHECKPOINT_GLOBAL(1, sem_id, barrier_coord_x, barrier_coord_y, num_cores);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        pack_tile(b, tt::CBIndex::c_16);
        cb_pop_front(tt::CBIndex::c_0, 1);
        cb_push_back(tt::CBIndex::c_16, 1);
    }

    release_dst();
}
