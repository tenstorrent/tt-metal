// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t n_tiles = get_arg(args::n_tiles);

    // We are going to read from these two dataflow buffers
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    // and write to the output dataflow buffer
    DataflowBuffer dfb_out0(dfb::out0);

    // The destination register is a set of 16 tiles. Which the matrix engine (FPU) can output
    // to. For our case, we are going perform the addition of two tiles and write the result
    // to destination register 0.
    constexpr uint32_t dst_reg = 0;

    // Tell the SFPU that we will be using dataflow buffers in0, in1 and out0
    // to perform the computation.
    compute_kernel_hw_startup(dfb::in0, dfb::in1, dfb::out0);
    // And we are going to add tiles. This function is only called if we ever need to
    // switch operation to something else. Since we are only adding tiles, this function
    // is only called once before the loop.
    add_init(dfb::in0, dfb::in1);

    // Loop over all the tiles and perform the computation
    for (uint32_t i = 0; i < n_tiles; i++) {
        // Wait until there is a tile in both input dataflow buffers
        dfb_in0.wait_front(1);
        dfb_in1.wait_front(1);
        // Make sure there is registers we can use and hold it. The register can be being used by other
        // components. So we need to be sure before we use it. Thus even though there is 16 registers, each
        // time acquire a register, we get 8 of them that we can use until released.
        tile_regs_acquire();
        // Add the tiles from the input dataflow buffers and write the result to the destination register
        add_tiles(dfb::in0, dfb::in1, 0, 0, dst_reg);
        // Release the held register
        tile_regs_commit();
        tile_regs_wait();
        // Make sure there is space in the output dataflow buffer
        dfb_out0.reserve_back(1);
        // Copy the result from adding the tiles to the output dataflow buffer
        pack_tile(dst_reg, dfb::out0);
        // Mark the output tile as ready and pop the input tiles
        dfb_out0.push_back(1);
        dfb_in0.pop_front(1);
        dfb_in1.pop_front(1);
        // Release the held register
        tile_regs_release();
    }
}
