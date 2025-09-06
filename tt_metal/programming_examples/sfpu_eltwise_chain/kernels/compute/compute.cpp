// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"

namespace NAMESPACE {
void MAIN {
    // Compile time args
    constexpr uint32_t src_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t ones_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t result_cb_index = get_compile_time_arg_val(2);

    // Constants
    constexpr uint32_t one_tile = 1;

    // Initialize the SFPU
    init_sfpu(src_cb_index, result_cb_index);

    // Wait for the SFPU to have registers available for us to use during
    // the computation.
    tile_regs_acquire();

    // Wait for data to show up in the circular buffer and copy it from
    // the circular buffer to registers so the SFPU can use it.
    // the first 0 in copy_tile is the index into the circular buffer
    // and the second 0 is the offset into the registers. This case
    // we are copying the 0th tile from the source data circular buffer to the 0th tile
    // in the registers and 0th tile from the ones tile to the 1st tile in the registers.
    cb_wait_front(src_cb_index, one_tile);
    cb_wait_front(ones_cb_index, one_tile);
    copy_tile(src_cb_index, /*offset*/ 0, /*register_offset*/ 0);
    copy_tile(ones_cb_index, /*offset*/ 0, /*register_offset*/ 1);

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

    // Wait for result to be done and data stored back to the circular buffer
    tile_regs_commit();
    tile_regs_wait();

    // Reserve output tile
    cb_reserve_back(result_cb_index, one_tile);

    pack_tile(0, result_cb_index);  // copy tile 0 from the registers to the CB

    // We don't need the input tile anymore, mark it as consumed
    cb_pop_front(src_cb_index, one_tile);
    cb_pop_front(ones_cb_index, one_tile);

    // Done with the registers, we can release them for the next SFPU operation
    tile_regs_release();

    // Mark the tile as ready for the writer kernel to write to DRAM
    cb_push_back(result_cb_index, one_tile);
}
}  // namespace NAMESPACE
