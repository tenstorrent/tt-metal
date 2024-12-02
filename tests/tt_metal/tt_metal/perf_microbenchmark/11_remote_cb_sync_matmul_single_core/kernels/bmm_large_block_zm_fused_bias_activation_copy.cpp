// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/matmul.h"

namespace NAMESPACE {

void MAIN {
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);  // inner block size in tiles
    constexpr uint32_t in0_block_num_tiles =
        get_compile_time_arg_val(1);  // out_subblock_h*in0_block_w*in0_num_subblocks;
    constexpr uint32_t in1_block_num_tiles =
        get_compile_time_arg_val(2);                                  // out_subblock_w*in0_block_w* in1_num_subblocks;
    constexpr uint32_t in1_per_core_w = get_compile_time_arg_val(3);  // out_subblock_w*in1_num_subblocks
    constexpr uint32_t num_blocks = get_compile_time_arg_val(4);      // outer inner dim (in inner dim blocks)
    constexpr uint32_t out_block_h = get_compile_time_arg_val(5);     // inner row block size in tiles
    constexpr uint32_t out_block_w = get_compile_time_arg_val(6);     // inner column block size in tiles
    constexpr uint32_t out_block_num_tiles = get_compile_time_arg_val(7);  // out_subblock_h * out_subblock_w;
    constexpr uint32_t num_layers = get_compile_time_arg_val(8);           // untilize output

    constexpr uint32_t in0_cb_id = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t sync_cb_id = tt::CBIndex::c_2;
    constexpr uint32_t out_cb_id = tt::CBIndex::c_16;

    for (uint32_t l = 0; l < num_layers; ++l) {
        mm_block_init(in0_cb_id, in1_cb_id, out_cb_id, false, out_block_w, out_block_h, in0_block_w);

        tile_regs_acquire();

        for (uint32_t block = 0; block < num_blocks; block++) {
            cb_wait_front(in0_cb_id, in0_block_num_tiles);
            cb_wait_front(in1_cb_id, in1_block_num_tiles);

            // Compute output sub-block
            uint32_t dst_index = 0;  // start at 0, each call to matmul_block internally increments dst_index
            uint32_t in0_index = 0;  // offset into in0 block
            uint32_t in1_index = 0;  // offset into in1 block
            // inner dim that we accumualte is the inner dim of in0/in1, which is in0_block_w
            for (uint32_t inner_dim_idx = 0; inner_dim_idx < in0_block_w; ++inner_dim_idx) {
                // matmul outer product of (out_subblock_h x out_subblock_w) tiles that fill dst
                // accumulation is done by iterating matmul_block across inner dim
                // in0_block_w is passed as innder dim (kt) to matmul_block, interally used to stride in0
                matmul_block(
                    in0_cb_id,
                    in1_cb_id,
                    in0_index,
                    in1_index,
                    dst_index,
                    false,
                    out_block_w,
                    out_block_h,
                    in0_block_w);
                in0_index++;                  // stride right by 1
                in1_index += in1_per_core_w;  // to stride down by 1 need to stride by in_per_core_w (should be called
                                              // in1_block_w)
            }

            cb_pop_front(in0_cb_id, in0_block_num_tiles);
            cb_pop_front(in1_cb_id, in1_block_num_tiles);

            // sync with the in1 receiver, so that the receiver knows when to pop the global CB
            cb_reserve_back(sync_cb_id, 1);
            cb_push_back(sync_cb_id, 1);
        }

        tile_regs_commit();
        // Pack out to output buffer
        cb_reserve_back(out_cb_id, out_block_num_tiles);
        tile_regs_wait();
        uint32_t start_dst_index = 0;
        matmul_pack_tile(start_dst_index, out_cb_id, out_block_num_tiles);
        tile_regs_release();
        cb_push_back(out_cb_id, out_block_num_tiles);
    }
}
}  // namespace NAMESPACE
