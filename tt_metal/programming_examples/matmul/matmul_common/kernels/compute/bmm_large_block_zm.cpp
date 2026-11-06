// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t in0_block_w = get_arg(args::in0_block_w);                  // inner block size in tiles
    uint32_t in0_num_subblocks = get_arg(args::in0_num_subblocks);      // outer row block size (in inner row blocks)
    uint32_t in0_block_num_tiles = get_arg(args::in0_block_num_tiles);  // out_subblock_h*in0_block_w*in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = get_arg(args::in0_subblock_num_tiles);  // out_subblock_h*in0_block_w
    uint32_t in1_num_subblocks = get_arg(args::in1_num_subblocks);  // outer column block size (in inner column blocks)
    uint32_t in1_block_num_tiles =
        get_arg(args::in1_block_num_tiles);                   // out_subblock_w*in0_block_w* in1_num_subblocks;
    uint32_t in1_per_core_w = get_arg(args::in1_per_core_w);  // out_subblock_w*in1_num_subblocks
    uint32_t num_blocks = get_arg(args::num_blocks);          // outer inner dim (in inner dim blocks)
    uint32_t out_subblock_h = get_arg(args::out_subblock_h);  // inner row block size in tiles
    uint32_t out_subblock_w = get_arg(args::out_subblock_w);  // inner column block size in tiles
    uint32_t out_subblock_num_tiles = get_arg(args::out_subblock_num_tiles);  // out_subblock_h * out_subblock_w;
    uint32_t batch = get_arg(args::batch);                                    // batch dim

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_out0(dfb::out0);
    // The intermediate buffer is both written and read by this kernel; it shares memory with the output buffer.
    DataflowBuffer dfb_interm0(dfb::interm0);

    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::in0, dfb::in1, dfb::out0);
    matmul_init(dfb::in0, dfb::in1);

    for (uint32_t b = 0; b < batch; b++) {
        bool spill = num_blocks > 1;
        bool enable_reload = false;
        uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;

        for (uint32_t block = 0; block < num_blocks; block++) {
            bool last_out = block == (num_blocks - 1);

            dfb_in0.wait_front(in0_block_num_tiles);
            dfb_in1.wait_front(in1_block_num_tiles);
            int in0_index_subblock_offset = 0;
            for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                int in1_index_subblock_offset = 0;
                for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                    tile_regs_acquire();

                    if (enable_reload) {
                        copy_init(dfb::interm0);
                        dfb_interm0.wait_front(out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            copy_tile(dfb::interm0, i, i);
                        }
                        dfb_interm0.pop_front(out_subblock_num_tiles);
                        matmul_init(dfb::in0, dfb::in1);
                    }

                    // Compute output sub-block from in0_subblock x in1_subblock
                    int dst_index = 0;
                    int in0_index_h_offset = 0;
                    for (uint32_t h = 0; h < out_subblock_h; h++) {
                        for (uint32_t w = 0; w < out_subblock_w; w++) {
                            int in1_index_inner_dim_offset = 0;
                            for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                                int in0_index = in0_index_subblock_offset + in0_index_h_offset + inner_dim;
                                int in1_index = in1_index_subblock_offset + in1_index_inner_dim_offset + w;
                                matmul_tiles(dfb::in0, dfb::in1, in0_index, in1_index, dst_index);
                                in1_index_inner_dim_offset += in1_per_core_w;
                            }
                            dst_index++;
                        }
                        in0_index_h_offset += in0_block_w;
                    }

                    tile_regs_commit();
                    tile_regs_wait();

                    if (last_out) {
                        // Pack out to output buffer
                        dfb_out0.reserve_back(out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            pack_tile(i, dfb::out0);
                        }
                        dfb_out0.push_back(out_subblock_num_tiles);
                    } else {
                        // Wait for tiles in output buffer to be written out since interm and output share memory
                        if (block == 0) {
                            dfb_out0.reserve_back(out_num_tiles_to_wait);
                            out_num_tiles_to_wait += out_subblock_num_tiles;
                        }
                        // Move partial result to interm buffer
                        dfb_interm0.reserve_back(out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            pack_tile(i, dfb::interm0);
                        }
                        dfb_interm0.push_back(out_subblock_num_tiles);
                    }

                    tile_regs_release();
                    in1_index_subblock_offset += out_subblock_w;
                }
                in0_index_subblock_offset += in0_subblock_num_tiles;
            }

            if (spill) {
                enable_reload = true;
            }

            dfb_in0.pop_front(in0_block_num_tiles);
            dfb_in1.pop_front(in1_block_num_tiles);
        }
    }
}
