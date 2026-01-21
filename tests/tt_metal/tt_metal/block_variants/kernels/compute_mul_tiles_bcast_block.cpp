// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

/**
 * Test Kernel: mul_tiles_bcast Block Operation
 *
 * Uses block variant to process multiple tiles at once.
 * Must produce identical results to tile-by-tile version.
 */

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/bcast.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    // Initialize operation (same as tile version)
    init_bcast<ELWMUL_TILES_BCAST, BroadcastType::COL>(cb_in0, cb_in1, cb_out);

    // Process blocks using block operation
    for (uint32_t block = 0; block < num_blocks; block++) {
        // Wait for input tiles
        cb_wait_front(cb_in0, Ht * Wt);
        cb_wait_front(cb_in1, Ht * Wt);

        // Acquire DEST
        tile_regs_acquire();

        // USE BLOCK OPERATION - This is what we're testing!
        mul_tiles_bcast_block<BroadcastType::COL, Ht, Wt>(cb_in0, cb_in1, 0, 0, 0);

        tile_regs_commit();

        // Pack results using pack_block
        cb_reserve_back(cb_out, Ht * Wt);
        tile_regs_wait();

        pack_block<Ht, Wt>(0, cb_out);

        tile_regs_release();

        // Push and pop
        cb_push_back(cb_out, Ht * Wt);
        cb_pop_front(cb_in0, Ht * Wt);
        cb_pop_front(cb_in1, Ht * Wt);
    }
}
}  // namespace NAMESPACE
