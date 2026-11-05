// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // Dataflow buffers
    DataflowBuffer src_dfb(dfb::src);
    DataflowBuffer ones_dfb(dfb::ones);
    DataflowBuffer result_dfb(dfb::result);

    // Constants
    constexpr uint32_t one_tile = 1;

    // Initialize the SFPU
    compute_kernel_hw_startup(dfb::src, dfb::result);
    copy_init(dfb::src);

    // Wait for the SFPU to have registers available for us to use during
    // the computation.
    tile_regs_acquire();

    // Wait for data to show up in the dataflow buffer and copy it from
    // the dataflow buffer to registers so the SFPU can use it.
    // the first 0 in copy_tile is the index into the dataflow buffer
    // and the second 0 is the offset into the registers. This case
    // we are copying the 0th tile from the source data dataflow buffer to the 0th tile
    // in the registers and 0th tile from the ones tile to the 1st tile in the registers.
    src_dfb.wait_front(one_tile);
    ones_dfb.wait_front(one_tile);
    copy_tile(dfb::src, /*offset*/ 0, /*register_offset*/ 0);
    copy_tile(dfb::ones, /*offset*/ 0, /*register_offset*/ 1);

    //
    // Fused operations
    //
    // Compute the softplus of the tile using the SFPU.
    // *_init() - Telling the SFPU to perform given operation. This is required each time we
    // switch to a different SFPU operation.
    exp_tile_init();
    exp_tile(0);  // exp(input)

    add_binary_tile_init();
    add_binary_tile(0, 1, 0);  // exp(input) + 1

    log_tile_init();
    log_tile(0);  // log(exp(input) + 1)

    // Wait for result to be done and data stored back to the dataflow buffer
    tile_regs_commit();
    tile_regs_wait();

    // Reserve output tile
    result_dfb.reserve_back(one_tile);

    pack_tile(0, dfb::result);  // copy tile 0 from the registers to the dataflow buffer

    // We don't need the input tile anymore, mark it as consumed
    src_dfb.pop_front(one_tile);
    ones_dfb.pop_front(one_tile);

    // Done with the registers, we can release them for the next SFPU operation
    tile_regs_release();

    // Mark the tile as ready for the writer kernel to write to DRAM
    result_dfb.push_back(one_tile);
}
