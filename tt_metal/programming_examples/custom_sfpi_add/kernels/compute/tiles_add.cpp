// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#ifdef TRISC_MATH
#include "experimental/llk_sfpu/ckernel_sfpu_custom_add.h"
#endif

/**
 * High-Level SFPU API Function
 *
 * This is the public interface that kernel code calls to perform custom SFPU operations.
 * It demonstrates the multi-layer abstraction in TT-Metal kernel programming:
 *
 * Abstraction Layers (from low to high):
 * 1. add_tile_face()         - Low-level SFPU function (operates on tile faces)
 * 2. my_add_tile()          - High-level API (easier to use in kernel code)
 *
 * The MATH() macro is crucial - it ensures this function only executes on cores
 * that is designated to the SFPU (vector engine). This is part of TT-Metal's
 * heterogeneous compute model where different cores have different capabilities.
 *
 * Usage Pattern:
 * - Call this function after loading data into Dst registers with copy_tile()
 * - Specify Dst register indices (not dataflow buffers!)
 * - Results are written back to the specified Dst register
 *
 * This function takes tile idx_dst0 and idx_dst1 as inputs, adds them,
 * and writes the result to tile idx_out0 in the Dst registers.
 */
inline void my_add_tile(uint32_t idx_dst0, uint32_t idx_dst1, uint32_t idx_out0) {
    MATH(SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, my_add_tile_face, idx_dst0, idx_dst1, idx_out0, VectorMode::RC));
}

void kernel_main() {
    uint32_t n_tiles = get_arg(args::n_tiles);

    // We are going to read from these two dataflow buffers
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    // and write to the output dataflow buffer
    DataflowBuffer dfb_out0(dfb::out0);

    // Tell the SFPU that we will be using dataflow buffers in0 and out0
    // to perform the computation.
    compute_kernel_hw_startup(dfb::in0, dfb::out0);
    copy_init(dfb::in0);

    // Loop over all the tiles and perform the computation
    for (uint32_t i = 0; i < n_tiles; i++) {
        // Wait until there is a tile in both input dataflow buffers
        dfb_in0.wait_front(1);
        dfb_in1.wait_front(1);
        // Make sure there is registers we can use and hold it. The register can be being used by other
        // components. So we need to be sure before we use it. Thus even though there is 16 registers, each
        // time acquire a register, we get 8 of them that we can use until released.
        tile_regs_acquire();
        // Copy the tiles from the dataflow buffers into Dst registers
        copy_tile(dfb::in0, 0, 0);
        copy_tile(dfb::in1, 0, 1);
        my_add_tile(0, 1, 0);  // <-- The custom SFPU addition happens here
        // Finished the computation, transfer register ownership to the unpacker
        tile_regs_commit();
        tile_regs_wait();
        // Make sure there is space in the output dataflow buffer
        dfb_out0.reserve_back(1);
        // Copy the result from adding the tiles to the output dataflow buffer
        pack_tile(0, dfb::out0);
        // Mark the output tile as ready and pop the input tiles
        dfb_out0.push_back(1);
        dfb_in0.pop_front(1);
        dfb_in1.pop_front(1);
        // Release the held register
        tile_regs_release();
    }
}
