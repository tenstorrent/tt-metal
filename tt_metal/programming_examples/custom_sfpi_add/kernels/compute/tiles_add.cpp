// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"

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
 * - Specify Dst register indices (not circular buffer indices!)
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
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

    // We are going to read from these two circular buffers
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    // and write to the output circular buffer
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    // Tell the SFPU that we will be using circular buffers c_in0 and c_out0
    // to perform the computation.
    init_sfpu(cb_in0, cb_out0);

    // Loop over all the tiles and perform the computation
    for (uint32_t i = 0; i < n_tiles; i++) {
        // Wait until there is a tile in both input circular buffers
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);
        // Make sure there is registers we can use and hold it. The register can be being used by other
        // components. So we need to be sure before we use it. Thus even though there is 16 registers, each
        // time acquire a register, we get 8 of them that we can use until released.
        tile_regs_acquire();
        // Copy the tiles from the circular buffers into Dst registers
        copy_tile(cb_in0, 0, 0);
        copy_tile(cb_in1, 0, 1);
        my_add_tile(0, 1, 0);  // <-- The custom SFPU addition happens here
        // Finished the computation, transfer register ownership to the unpacker
        tile_regs_commit();
        tile_regs_wait();
        // Make sure there is space in the output circular buffer
        cb_reserve_back(cb_out0, 1);
        // Copy the result from adding the tiles to the output circular buffer
        pack_tile(0, cb_out0);
        // Mark the output tile as ready and pop the input tiles
        cb_push_back(cb_out0, 1);
        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
        // Release the held register
        tile_regs_release();
    }
}
