// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"

#ifdef FUSE_BIAS
#include "compute_kernel_api/bcast.h"
#endif

#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

namespace NAMESPACE {
void MAIN {
    uint32_t in0_block_w = get_compile_time_arg_val(0);              // inner block size in tiles
    uint32_t in0_num_subblocks = get_compile_time_arg_val(1);        // outer row block size (in inner row blocks)
    uint32_t in0_block_num_tiles = get_compile_time_arg_val(2);      // out_subblock_h*in0_block_w*in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);   // out_subblock_h*in0_block_w
    uint32_t in1_num_subblocks = get_compile_time_arg_val(4);        // outer column block size (in inner column blocks)
    uint32_t in1_block_num_tiles = get_compile_time_arg_val(5);      // out_subblock_w*in0_block_w* in1_num_subblocks;
    uint32_t in1_per_core_w = get_compile_time_arg_val(6);           // out_subblock_w*in1_num_subblocks
    uint32_t num_blocks = get_compile_time_arg_val(7);               // outer inner dim (in inner dim blocks)
    uint32_t out_subblock_h = get_compile_time_arg_val(8);           // inner row block size in tiles
    uint32_t out_subblock_w = get_compile_time_arg_val(9);           // inner column block size in tiles
    uint32_t out_subblock_num_tiles = get_compile_time_arg_val(10);  // out_subblock_h * out_subblock_w;
    uint32_t batch = get_compile_time_arg_val(11);                   // batch dim

    uint32_t in0_cb_id = tt::CBIndex::c_0;
    uint32_t in1_cb_id = tt::CBIndex::c_1;
    uint32_t out_cb_id = tt::CBIndex::c_16;
    uint32_t mm_partials_cb_id = tt::CBIndex::c_24;
    uint32_t mm_bias_intermediate_cb_id = tt::CBIndex::c_25;
    uint32_t bias_cb_id = tt::CBIndex::c_3;

#ifdef FUSE_BIAS
    init_bcast<EltwiseBinaryType::ELWADD, BroadcastType::ROW>(mm_bias_intermediate_cb_id, bias_cb_id, out_cb_id);
#endif

    mm_init(in0_cb_id, in1_cb_id, out_cb_id);

    for (uint32_t b = 0; b < batch; b++) {
        bool spill = num_blocks > 1;
        bool enable_reload = false;
        uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;

        for (uint32_t block = 0; block < num_blocks; block++) {
            bool last_out = block == (num_blocks - 1);

            cb_wait_front(in0_cb_id, in0_block_num_tiles);
            cb_wait_front(in1_cb_id, in1_block_num_tiles);
            int in0_index_subblock_offset = 0;
            for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                int in1_index_subblock_offset = 0;
                for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                    acquire_dst();

                    if (enable_reload) {
                        // Reconfigure input
                        copy_tile_to_dst_init_short_with_dt(in1_cb_id, mm_partials_cb_id);
                        cb_wait_front(mm_partials_cb_id, out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            copy_tile(mm_partials_cb_id, i, i);
                        }
                        cb_pop_front(mm_partials_cb_id, out_subblock_num_tiles);
                        // Reconfigure srcA back
                        mm_init_short_with_dt(in0_cb_id, in1_cb_id, mm_partials_cb_id);
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
                                matmul_tiles(
                                    in0_cb_id, in1_cb_id, in0_index, in1_index, dst_index, false /* transpose */);
                                in1_index_inner_dim_offset += in1_per_core_w;
                            }
                            dst_index++;
                        }
                        in0_index_h_offset += in0_block_w;
                    }

                    if (last_out) {
#ifdef FUSE_BIAS
                        // Move matmul result to interm buffer
                        cb_reserve_back(mm_bias_intermediate_cb_id, out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            pack_tile(i, mm_bias_intermediate_cb_id);
                        }
                        cb_push_back(mm_bias_intermediate_cb_id, out_subblock_num_tiles);
                        release_dst();

                        // Redundant wait since we know data was just pushed
                        cb_wait_front(mm_bias_intermediate_cb_id, out_subblock_num_tiles);
                        cb_wait_front(bias_cb_id, in1_per_core_w);
                        add_bcast_rows_init_short();
                        // reconfigure unpacker df for src B
                        reconfig_data_format(mm_bias_intermediate_cb_id, bias_cb_id);
                        // reconfigure packer df for out
                        pack_reconfig_data_format(out_cb_id);
                        acquire_dst();
                        for (uint32_t i = 0, j = 0; j < out_subblock_h; j++) {
                            uint32_t bcast_tile_idx = in1_index_subblock_offset;
                            for (uint32_t k = 0; k < out_subblock_w; k++, i++) {
                                add_tiles_bcast_rows(mm_bias_intermediate_cb_id, bias_cb_id, i, bcast_tile_idx, i);
                                bcast_tile_idx++;
                            }
                        }
                        cb_pop_front(mm_bias_intermediate_cb_id, out_subblock_num_tiles);
                        // reconfigure init for matmul
                        mm_init_short();
                        // reconfigure unpacker df for src B
                        reconfig_data_format(in1_cb_id, in0_cb_id);
#endif

                        // sfpu activation
#ifdef SFPU_OP_INIT_ACTIVATION
                        SFPU_OP_INIT_ACTIVATION
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            SFPU_OP_FUNC_ACTIVATION
                        }
#endif
                        // Pack out to output buffer
                        cb_reserve_back(out_cb_id, out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            pack_tile(i, out_cb_id);
                        }
                        cb_push_back(out_cb_id, out_subblock_num_tiles);
                    } else {
                        // Wait for tiles in output buffer to be written out since interm and output share memory
                        if (block == 0) {
                            cb_reserve_back(out_cb_id, out_num_tiles_to_wait);
                            out_num_tiles_to_wait += out_subblock_num_tiles;
                        }
                        // Move partial result to interm buffer
                        cb_reserve_back(mm_partials_cb_id, out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            pack_tile(i, mm_partials_cb_id);
                        }
                        cb_push_back(mm_partials_cb_id, out_subblock_num_tiles);
                    }

                    release_dst();
                    in1_index_subblock_offset += out_subblock_w;
                }
                in0_index_subblock_offset += in0_subblock_num_tiles;
            }

            if (spill) {
                enable_reload = true;
            }

            cb_pop_front(in0_cb_id, in0_block_num_tiles);
            cb_pop_front(in1_cb_id, in1_block_num_tiles);
        }
    }
}
}  // namespace NAMESPACE
