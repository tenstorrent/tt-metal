// SPDX-FileCopyrightText: © 2024 Martin Chang
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
    constexpr auto cb_in0 = tt::CB::c_in0;
    constexpr auto cb_in1 = tt::CB::c_in1;
    // and write to the output circular buffer
    constexpr auto cb_out0 =  tt::CB::c_out0;
    // The destination register.
    // Quote the doc: "This register is an array of 16 tiles of 32x32 elements each."
    // If you are fimilar with the concept of rotating register file from computer
    // architecture. Think it like that. Later on we will ensure that registers are
    // free and then we will submit compute to the FPU/SFPU that writes to the register.
    // see: https://tenstorrent-metal.github.io/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/compute/acquire_dst.html
    constexpr uint32_t dst_reg = 0;

    // Tell the SFPU that we will be using circular buffers c_in0, c_in1 and c_out0
    // to perform the computation.
    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    // And we are going to add tiles. This function is only called if we ever need to
    // switch operation to something else. Since we are only adding tiles, this function
    // is only called once before the loop.
    add_tiles_init();

    // Loop over all the tiles and perform the computation
    for(uint32_t i = 0; i < n_tiles; i++) {
        // Make sure there is a valid register we can use.
        acquire_dst(tt::DstMode::Half);
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
        release_dst(tt::DstMode::Half);
    }
}
}
