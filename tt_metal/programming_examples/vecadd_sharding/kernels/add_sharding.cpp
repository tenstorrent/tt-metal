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
    // IMPORTANT: since there is no read kernel, and data is alraedy in circular buffers
    // do not call cb_wait_front() because there is no wait. And we ensured there is enough
    // spece in the circular buffers for the entirty of the computation.
    // if calling cb_wait_front() here, the kernel will hang forever as no one is producing
    // data to the circular buffers.

    // We are going to read from these two circular buffers
    constexpr auto cb_in0 = get_compile_time_arg_val(0);
    constexpr auto cb_in1 = get_compile_time_arg_val(1);
    // and write to the output circular buffer
    constexpr auto cb_out0 = get_compile_time_arg_val(2);

    uint32_t num_tile = get_arg_val<uint32_t>(0);

    constexpr uint32_t dst_reg = 0;

    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    add_tiles_init(cb_in0, cb_in1);

    // The standard vector addition kernel but without waiting on circular buffers (as explained
    // above).
    for (uint32_t i = 0; i < num_tile; i++) {
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
