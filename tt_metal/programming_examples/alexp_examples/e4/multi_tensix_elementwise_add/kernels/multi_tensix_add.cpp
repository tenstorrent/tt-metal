// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"

namespace NAMESPACE {
void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);  // Number of A tiles for this Tensix core
    uint32_t r_tiles = get_arg_val<uint32_t>(1);  // Number of B tiles (replicated)

    constexpr auto cb_in0 = tt::CB::c_in0;      // A tiles (distributed)
    constexpr auto cb_in1 = tt::CB::c_in1;      // B tiles (replicated)
    constexpr auto cb_interm = tt::CB::c_in2;   // Intermediate results
    constexpr auto cb_out = tt::CB::c_out0;     // Final output

    binary_op_init_common(cb_in0, cb_in1, cb_interm);
    add_tiles_init();

    for (uint32_t i = 0; i < n_tiles; i++) {
        // Get A tile
        cb_wait_front(cb_in0, 1);

        // Initialize intermediate result with first B tile + A tile
        cb_wait_front(cb_in1, 1);
        cb_reserve_back(cb_interm, 1);

        tile_regs_acquire();
        add_tiles(cb_in0, cb_in1, 0, 0, 0);
        tile_regs_commit();

        cb_push_back(cb_interm, 1);
        cb_pop_front(cb_in1, 1);
        tile_regs_wait();
        tile_regs_release();

        // Add remaining B tiles to intermediate result
        for (uint32_t j = 1; j < r_tiles; j++) {
            cb_wait_front(cb_in1, 1);
            cb_wait_front(cb_interm, 1);
            cb_reserve_back(cb_interm, 1);

            tile_regs_acquire();
            add_tiles(cb_interm, cb_in1, 0, 0, 0);
            tile_regs_commit();

            cb_push_back(cb_interm, 1);
            cb_pop_front(cb_in1, 1);
            cb_pop_front(cb_interm, 1);
            tile_regs_wait();
            tile_regs_release();
        }

        // Move final result to output
        cb_wait_front(cb_interm, 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();
        copy_tile(cb_interm, 0, 0);
        tile_regs_commit();

        cb_push_back(cb_out, 1);
        cb_pop_front(cb_interm, 1);
        cb_pop_front(cb_in0, 1);
        tile_regs_wait();
        tile_regs_release();
    }
}
}  // namespace NAMESPACE
