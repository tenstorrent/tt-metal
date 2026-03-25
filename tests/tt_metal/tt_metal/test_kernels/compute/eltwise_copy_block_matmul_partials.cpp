// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
// #include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"
// #include "api/compute/compute_kernel_api.h"
#ifdef ARCH_QUASAR
#include "experimental/dataflow_buffer.h"
#else
#include "experimental/circular_buffer.h"
#endif

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t num_single_transfer = get_compile_time_arg_val(1);
    constexpr uint32_t in_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(3);

    constexpr uint32_t outer_loop = num_tiles / num_single_transfer;

#ifdef ARCH_QUASAR
    experimental::DataflowBuffer dfb_in(in_cb_id);
    experimental::DataflowBuffer dfb_out(out_cb_id);
    unary_op_init_common(dfb_in.get_id(), dfb_out.get_id());
#else
    experimental::CircularBuffer cb_in(in_cb_id);
    experimental::CircularBuffer cb_out(out_cb_id);
    unary_op_init_common(in_cb_id, out_cb_id);
#endif

    // Run the outer loop
    for (uint32_t b = 0; b < outer_loop; ++b) {
#ifdef ARCH_QUASAR
        dfb_in.wait_front(num_single_transfer);
        acquire_dst();
        dfb_out.reserve_back(num_single_transfer);

        for (uint32_t i = 0; i < num_single_transfer; ++i) {
            copy_block_matmul_partials(dfb_in.get_id(), i, i, 1);
        }

        pack_tile_block(0, dfb_out.get_id(), num_single_transfer);

        release_dst();
        dfb_in.pop_front(num_single_transfer);
        dfb_out.push_back(num_single_transfer);
#else
        // Wait for num_single_transfer tiles to be available in in_cb
        cb_in.wait_front(num_single_transfer);
        // Acquire DEST reg for MATH/PACK
        acquire_dst();
        // Reserve out_cb space for num_single_transfer tiles
        cb_out.reserve_back(num_single_transfer);

        // Copy num_single_transfer tiles from in_cb to DEST
        for (uint32_t i = 0; i < num_single_transfer; ++i) {
            copy_block_matmul_partials(in_cb_id, i, i, 1);
        }
        // Pack num_single_transfer tiles to out_cb
        pack_tile_block(0, out_cb_id, num_single_transfer);

        // Release DEST reg marking compute/pack complete
        release_dst();
        // Move rd ptr from in_cb by num_single_transfer places
        cb_in.pop_front(num_single_transfer);
        // Move wr prt from out_cb by num_single_transfer places
        cb_out.push_back(num_single_transfer);
#endif
    }
}
