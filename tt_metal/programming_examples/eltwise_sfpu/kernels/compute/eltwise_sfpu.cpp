// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t n_tiles = get_arg(args::n_tiles);

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_out0(dfb::out0);

    // Initialize the SFPU
    compute_kernel_hw_startup(dfb::in0, dfb::out0);
    copy_init(dfb::in0);
    // Telling the SFPU to perform exponential. This is required each time we
    // switch to a different SFPU operation.
    exp_tile_init();
    for (uint32_t i = 0; i < n_tiles; i++) {
        // Wait for the SFPU to have registers available for us to use during
        // the computation.
        tile_regs_acquire();
        // Wait for data to show up in the dataflow buffer and copy it from
        // the dataflow buffer to registers so the SFPU can use it.
        // the first 0 in copy_tile is the index into the dataflow buffer
        // and the second 0 is the offset into the registers. This case
        // we are copying the 0th tile from the dataflow buffer to the 0th tile
        // in the registers.
        dfb_in0.wait_front(1);
        copy_tile(dfb::in0, /*offset*/ 0, /*register_offset*/ 0);
        exp_tile(0);  // Compute the exponential of the tile using the SFPU. This
                      // operation is in-place. It takes data from tile 0 in the
                      // registers and writes the result back to tile 0 in the
                      // registers.
        // Wait for result to be done and data stored back to the dataflow buffer
        tile_regs_commit();
        tile_regs_wait();
        // Wait for space in the dataflow buffer to be available for us to write
        dfb_out0.reserve_back(1);
        pack_tile(0, dfb::out0);  // copy tile 0 from the registers to the dataflow buffer
        // We don't need the input tile anymore, mark it as consumed
        dfb_in0.pop_front(1);
        // Done with the registers, we can release them for the next SFPU operation
        tile_regs_release();
        // Mark the tile as ready for the writer kernel to write to DRAM
        dfb_out0.push_back(1);
    }
}
