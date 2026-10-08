// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Round 3 eltwise binary twin of deepseek_v3_b1 reduce_to_one's compute (TRISC) block
// (models/demos/deepseek_v3_b1/unified_kernels/reduce_to_one_b1.hpp:586-645, after reduce_to_one_kernel.cpp:93's
// deepseek_compute_kernel_init): the same calls for a ROOT3, ROOT2 or ROOT1 device (1, 2 or 3 dest-reuse accumulations),
// run TWIN_ITERS times. Compile args: num_tiles, local_cb, received_cb, scratch_cb, rounds, iterations.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/experimental/pack_block.h"
#include "api/compute/experimental/deepseek_compute_kernel_hw_startup.h"

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t local_cb = get_compile_time_arg_val(1);
    constexpr uint32_t received_cb = get_compile_time_arg_val(2);
    constexpr uint32_t scratch_cb = get_compile_time_arg_val(3);
    constexpr uint32_t rounds = get_compile_time_arg_val(4);
    constexpr uint32_t twin_iters = get_compile_time_arg_val(5);

    deepseek_compute_kernel_init();

    for (uint32_t it = 0; it < twin_iters; ++it) {
        // Initialize for binary operations
        reconfig_full_operand(local_cb, received_cb);
        pack_reconfig_data_format<true>(scratch_cb);
        pack_block_contiguous_init(scratch_cb);

        // Load local tiles to dest
        copy_init(local_cb);
        cb_wait_front(local_cb, num_tiles);
        tile_regs_acquire();
        for (uint32_t i = 0; i < num_tiles; i++) {
            copy_tile(local_cb, i, i);
        }
        cb_pop_front(local_cb, num_tiles);

        // Accumulate from received_cb page 0 (LEAF data)
        add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(received_cb);
        cb_wait_front(received_cb, num_tiles);
        for (uint32_t i = 0; i < num_tiles; i++) {
            add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(received_cb, i, i);
        }
        cb_pop_front(received_cb, num_tiles);

        if constexpr (rounds >= 2) {
            // Accumulate from received_cb page 1 (ROOT3 data)
            cb_wait_front(received_cb, num_tiles);
            for (uint32_t i = 0; i < num_tiles; i++) {
                add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(received_cb, i, i);
            }
            cb_pop_front(received_cb, num_tiles);
        }

        if constexpr (rounds >= 3) {
            // Accumulate from received_cb page 2 (ROOT2 data)
            cb_wait_front(received_cb, num_tiles);
            for (uint32_t i = 0; i < num_tiles; i++) {
                add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(received_cb, i, i);
            }
            cb_pop_front(received_cb, num_tiles);
        }
        tile_regs_commit();

        // Pack result to scratch_cb
        cb_reserve_back(scratch_cb, num_tiles);
        tile_regs_wait();
        pack_block_contiguous(0, scratch_cb, num_tiles);
        tile_regs_release();
        cb_push_back(scratch_cb, num_tiles);
    }
}
