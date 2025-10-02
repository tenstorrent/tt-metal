// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api.h"

#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"

namespace NAMESPACE {
void MAIN {
    // Runtime arguments
    uint32_t core_num_tiles = get_arg_val<uint32_t>(0);  // Number of tiles for this core
    uint32_t r_tiles = get_arg_val<uint32_t>(1);         // Number of B tiles (replicated)

    constexpr auto cb_in0 = tt::CBIndex::c_0;      // A tiles (distributed)
    constexpr auto cb_in1 = tt::CBIndex::c_1;      // B tiles (replicated)
    constexpr auto cb_interm = tt::CBIndex::c_2;   // Intermediate results
    constexpr auto cb_out0 = tt::CBIndex::c_16;    // Final output

    constexpr uint32_t acc_reg = 0;
    constexpr uint32_t b_reg = 1;

    init_sfpu(cb_in0, cb_out0);
    add_binary_tile_init();

    // Process each A tile assigned to this Tensix core
    for (uint32_t i = 0; i < core_num_tiles; i++) {
        // Get A tile for this core
        cb_wait_front(cb_in0, 1);

        // Initialize accumulator with A tile
        tile_regs_acquire();
        copy_tile(cb_in0, 0, acc_reg);
        tile_regs_commit();
        tile_regs_wait();
        tile_regs_release();

        cb_pop_front(cb_in0, 1);

        // Add all B tiles to the accumulator
        for (uint32_t j = 0; j < r_tiles; j++) {
            cb_wait_front(cb_in1, 1);

            tile_regs_acquire();
            copy_tile(cb_in1, 0, b_reg);
            add_binary_tile(acc_reg, b_reg, acc_reg);
            tile_regs_commit();
            tile_regs_wait();
            tile_regs_release();

            cb_pop_front(cb_in1, 1);
        }

        // Output final result
        cb_reserve_back(cb_out0, 1);

        tile_regs_acquire();
        pack_tile(acc_reg, cb_out0);
        tile_regs_commit();
        tile_regs_wait();
        tile_regs_release();

        cb_push_back(cb_out0, 1);
    }
}
}  // namespace NAMESPACE
