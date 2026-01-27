// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

/**
 * Reference Kernel: reduce Tile-by-Tile
 *
 * Processes blocks tile-by-tile as a reference implementation.
 * Used to validate block variant correctness.
 */

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/reduce_custom.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    // Initialize operation
    reduce_init<REDUCE_OP, REDUCE_COL>(cb_in, cb_scaler, cb_out);

    // Process blocks tile-by-tile
    for (uint32_t block = 0; block < num_blocks; block++) {
        // Wait for input tiles
        cb_wait_front(cb_in0, Ht * Wt);
        cb_wait_front(cb_in1, Ht * Wt);

        // Acquire DEST
        tile_regs_acquire();

        // Process tile-by-tile
        for (uint32_t h = 0; h < Ht; h++) {
            for (uint32_t w = 0; w < Wt; w++) {
                uint32_t tile_idx = h * Wt + w;
                reduce_tile<REDUCE_OP, REDUCE_COL>(cb_in, cb_scaler, tile_idx, tile_idx, tile_idx);
            }
        }

        tile_regs_commit();

        // Pack results
        cb_reserve_back(cb_out, Ht * Wt);
        tile_regs_wait();

        for (uint32_t i = 0; i < Ht * Wt; i++) {
            pack_tile(i, cb_out);
        }

        tile_regs_release();

        // Push and pop
        cb_push_back(cb_out, Ht * Wt);
        cb_pop_front(cb_in0, Ht * Wt);
        cb_pop_front(cb_in1, Ht * Wt);
    }
}
}  // namespace NAMESPACE
