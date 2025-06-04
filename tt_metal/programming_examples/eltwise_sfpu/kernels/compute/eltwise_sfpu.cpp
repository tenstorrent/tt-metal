// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"

namespace NAMESPACE {
void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

    // Initialize the SFPU
    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_16);
    // Telling the SFPU to perform exponential. This is required each time we
    // switch to a different SFPU operation.
    exp_tile_init();
    for (uint32_t i = 0; i < n_tiles; i++) {
        // Wait for the SFPU to have registers available for us to use during
        // the computation.
        tile_regs_acquire();
        // Wait for data to show up in the circular buffer and copy it from
        // the circular buffer to registers so the SFPU can use it.
        // the first 0 in copy_tile is the index into the circular buffer
        // and the second 0 is the offset into the registers. This case
        // we are copying the 0th tile from the circular buffer to the 0th tile
        // in the registers.
        cb_wait_front(tt::CBIndex::c_0, 1);
        copy_tile(tt::CBIndex::c_0, /*offset*/ 0, /*register_offset*/ 0);
        exp_tile(0);  // Compute the exponential of the tile using the SFPU. This
                      // operation is in-place. It takes data from tile 0 in the
                      // registers and writes the result back to tile 0 in the
                      // registers.
        // Wait for result to be done and data stored back to the circular buffer
        tile_regs_commit();
        tile_regs_wait();
        // Wait for space in the circular buffer to be available for us to write
        cb_reserve_back(tt::CBIndex::c_16, 1);
        pack_tile(0, tt::CBIndex::c_16);  // copy tile 0 from the registers to the CB
        // We don't need the input tile anymore, mark it as consumed
        cb_pop_front(tt::CBIndex::c_0, 1);
        // Done with the registers, we can release them for the next SFPU operation
        tile_regs_release();
        // Mark the tile as ready for the writer kernel to write to DRAM
        cb_push_back(tt::CBIndex::c_16, 1);
    }
}
}  // namespace NAMESPACE
