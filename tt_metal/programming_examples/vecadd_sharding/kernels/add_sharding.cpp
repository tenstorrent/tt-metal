// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include <cstdint>

namespace NAMESPACE {
void MAIN {
    // We are going to read from these two circular buffers
    constexpr auto cb_in0 = get_compile_time_arg_val(0);
    constexpr auto cb_in1 = get_compile_time_arg_val(1);
    // and write to the output circular buffer
    constexpr auto cb_out0 = get_compile_time_arg_val(2);

    uint32_t num_tile = get_arg_val<uint32_t>(0);

    // The destination register.
    // Quote the doc: "This register is an array of 16 tiles of 32x32 elements
    // each." If you are familiar with the concept of rotating register file
    // from computer architecture. Think it like that. Later on we will ensure
    // that registers are free and then we will submit compute to the FPU/SFPU
    // that writes to the register.
    constexpr uint32_t dst_reg = 0;

    // Tell the SFPU that we will be using circular buffers c_in0, c_in1 and
    // c_out0 to perform the computation.
    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    // And we are going to add tiles. This function is only called if we ever
    // need to switch operation to something else. Since we are only adding
    // tiles, this function is only called once before the loop.
    add_tiles_init(cb_in0, cb_in1);

    // Loop over the assigned tiles and perform the computation
    for (uint32_t i = 0; i < num_tile; i++) {
        // IMPORTANT: since there is no read kernel, and data is alraedy in circular buffers
        // do not call cb_wait_front() because there is no wait.
        // if calling cb_wait_front() here, the kernel will hang forever.

        // Make sure there is a valid MATH thread register we can use.
        tile_regs_acquire();

        // Add the tiles from the input circular buffers and write the result to
        // the destination register
        add_tiles(cb_in0, cb_in1, 0, 0, dst_reg);

        // release lock on DST register by MATH thread
        tile_regs_commit();

        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);

        // acquire an exclusive lock on the DST register for the PACK thread.
        // make sure MATH thread has committed the DST register earlier
        tile_regs_wait();

        //  Copy the result from adding the tiles to the output circular buffer
        pack_tile(dst_reg, cb_out0);

        // release lock on DST register by PACK thread
        tile_regs_release();

        // no need to call cb_reserve_back(cb_out0, 1)
        // buffer because output circular buffer is pointed to already allocated L1 buffer
        // but it does not hurt to call it

        // Mark the output tile as ready and pop the input tiles
        cb_push_back(cb_out0, 1);
    }
}
}  // namespace NAMESPACE
