// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

    // We are going to read from these two circular buffers
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    // and write to the output circular buffer
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    // The destination register is a set of 16 tiles. Which the matrix engine (FPU) can output
    // to. For our case, we are going perform the addition of two tiles and write the result
    // to destination register 0.
    constexpr uint32_t dst_reg = 0;

    // Tell the SFPU that we will be using circular buffers c_in0, c_in1 and c_out0
    // to perform the computation.
    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    // And we are going to add tiles. This function is only called if we ever need to
    // switch operation to something else. Since we are only adding tiles, this function
    // is only called once before the loop.
    add_tiles_init(cb_in0, cb_in1);

    // Loop over all the tiles and perform the computation
    for (uint32_t i = 0; i < n_tiles; i++) {
        // Make sure there is registers we can use and hold it. The register can be being used by other
        // components. So we need to be sure before we use it. Thus even though there is 16 registers, each
        // time acquire a register, we get 8 of them that we can use until released.
        acquire_dst();
        // Wait until there is a tile in both input circular buffers
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);
        // Add the tiles from the input circular buffers and write the result to the destination register
        add_tiles(cb_in0, cb_in1, 0, 0, dst_reg);
        // Make sure there is space in the output circular buffer
        cb_reserve_back(cb_out0, 1);
        // Copy the result from adding the tiles to the output circular buffer
        pack_tile(dst_reg, cb_out0);
        // Mark the output tile as ready and pop the input tiles
        cb_push_back(cb_out0, 1);
        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
        // Release the held register
        release_dst();
    }
}
}  // namespace NAMESPACE
