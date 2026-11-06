// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include <cstdint>

void kernel_main() {
    // IMPORTANT: since there is no read kernel, and data is already in dataflow buffers
    // do not call wait_front() because there is no wait. And we ensured there is enough
    // space in the dataflow buffers for the entirty of the computation.
    // if calling wait_front() here, the kernel will hang forever as no one is producing
    // data to the dataflow buffers.

    // We are going to read from these two dataflow buffers
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    // and write to the output dataflow buffer
    DataflowBuffer dfb_out0(dfb::out0);

    uint32_t num_tile = get_arg(args::num_tile);

    constexpr uint32_t dst_reg = 0;

    compute_kernel_hw_startup(dfb::in0, dfb::in1, dfb::out0);
    add_init(dfb::in0, dfb::in1);

    // The standard vector addition kernel but without waiting on dataflow buffers (as explained
    // above).
    for (uint32_t i = 0; i < num_tile; i++) {
        tile_regs_acquire();

        // Add the tiles from the input dataflow buffers and write the result to
        // the destination register
        add_tiles(dfb::in0, dfb::in1, 0, 0, dst_reg);

        // release lock on DST register by MATH thread
        tile_regs_commit();

        dfb_in0.pop_front(1);
        dfb_in1.pop_front(1);

        // acquire an exclusive lock on the DST register for the PACK thread.
        // make sure MATH thread has committed the DST register earlier
        tile_regs_wait();

        //  Copy the result from adding the tiles to the output dataflow buffer
        pack_tile(dst_reg, dfb::out0);

        // release lock on DST register by PACK thread
        tile_regs_release();

        // no need to call reserve_back(1)
        // buffer because output dataflow buffer is pointed to already allocated L1 buffer
        // but it does not hurt to call it

        // Mark the output tile as ready and pop the input tiles
        dfb_out0.push_back(1);
    }
}
