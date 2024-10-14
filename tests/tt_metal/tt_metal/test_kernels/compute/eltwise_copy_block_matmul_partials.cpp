// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {

    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t num_single_transfer = get_compile_time_arg_val(1);
    constexpr uint32_t in_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(3);

    constexpr uint32_t outer_loop = num_tiles / num_single_transfer;

    unary_op_init_common(in_cb_id, out_cb_id);

    // Run the outer loop
    for(uint32_t b = 0; b < outer_loop; ++b) {
        // Wait for num_single_transfer tiles to be available in in_cb
        cb_wait_front(in_cb_id, num_single_transfer);
        // Acquire DEST reg for MATH/PACK
        acquire_dst();
        // Reserve out_cb space for num_single_transfer tiles
        cb_reserve_back(out_cb_id, num_single_transfer);

        // Copy num_single_transfer tiles from in_cb to DEST
        copy_block_matmul_partials(in_cb_id, 0, 0, num_single_transfer);
        // Pack num_single_transfer tiles to out_cb
        matmul_pack_tile(0, out_cb_id, num_single_transfer);

        // Release DEST reg marking compute/pack complete
        release_dst();
        // Move rd ptr from in_cb by num_single_transfer places
        cb_pop_front(in_cb_id, num_single_transfer);
        // Move wr prt from out_cb by num_single_transfer places
        cb_push_back(out_cb_id, num_single_transfer);
    }
}
}
