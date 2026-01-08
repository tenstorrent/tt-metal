// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Kernel: Compute Add
 *
 * Performs element-wise addition of tiles from two input circular buffers
 * and writes result to output circular buffer.
 *
 * Runtime args:
 *   - n_tiles: number of tiles to process (0 for non-workers)
 *
 * CB layout:
 *   - CB 0: Input A tiles
 *   - CB 1: Input B tiles
 *   - CB 2: Output tiles (A + B)
 */

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
    // Get number of tiles to process from runtime args
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

    // Early exit for non-workers
    if (n_tiles == 0) {
        return;
    }

    // Circular buffer indices
    constexpr auto cb_in0 = tt::CBIndex::c_0;  // Input A
    constexpr auto cb_in1 = tt::CBIndex::c_1;  // Input B
    constexpr auto cb_out = tt::CBIndex::c_2;  // Output

    // Destination register index
    constexpr uint32_t dst_reg = 0;

    // Initialize binary operation for add
    binary_op_init_common(cb_in0, cb_in1, cb_out);
    add_tiles_init(cb_in0, cb_in1);

    // Process all tiles
    for (uint32_t i = 0; i < n_tiles; i++) {
        // Wait for input tiles from dataflow kernel
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);

        // Acquire destination register
        tile_regs_acquire();

        // Perform element-wise addition: C = A + B
        add_tiles(cb_in0, cb_in1, 0, 0, dst_reg);

        // Commit result to destination register
        tile_regs_commit();

        // Reserve space in output CB
        cb_reserve_back(cb_out, 1);

        // Wait for result and pack to output CB
        tile_regs_wait();
        pack_tile(dst_reg, cb_out);
        tile_regs_release();

        // Push output tile and pop input tiles
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
    }
}
}  // namespace NAMESPACE
