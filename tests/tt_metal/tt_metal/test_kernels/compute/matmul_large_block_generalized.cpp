// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tilize_untilize.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"


inline void tilize_activation(uint32_t in0_cb, uint32_t in0_subblock_h, uint32_t in0_block_w, uint32_t in0_num_subblocks, uint32_t out_cb)
{
    tilize_init_short(in0_cb, in0_block_w);

    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
        for (uint32_t h = 0; h < in0_subblock_h; h++) {
            cb_wait_front(in0_cb, in0_block_w);
            cb_reserve_back(out_cb, in0_block_w);
            tilize_block(in0_cb, in0_block_w, out_cb);
            cb_push_back(out_cb, in0_block_w);
            cb_pop_front(in0_cb, in0_block_w);
        }
    }

    tilize_uninit(in0_cb);

}

inline void reblock_and_untilize(
    uint32_t num_out_subblocks_in_col,
    uint32_t out_subblock_num_tiles,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t out_block_w,
    uint32_t interm_cb_id,
    uint32_t reblock_cb_id,
    uint32_t out_cb_id)
{
    uint32_t num_tiles_in_row_of_subblocks = mulsi3(out_subblock_num_tiles, num_out_subblocks_in_col);
    cb_wait_front(interm_cb_id, num_tiles_in_row_of_subblocks);

    int within_block_index = 0;
    for (uint32_t h = 0; h < out_subblock_h; h++) {
        int block_offset = 0;

        // Reblock
        copy_tile_to_dst_init_short();
        cb_reserve_back(reblock_cb_id, out_block_w);
        for (uint32_t n = 0; n < num_out_subblocks_in_col; n++) {
            for (uint32_t w = 0; w < out_subblock_w; w++) {
                uint32_t tile_index = block_offset + within_block_index + w;
                acquire_dst();
                copy_tile(interm_cb_id, tile_index, 0);
                pack_tile(0, reblock_cb_id);
                release_dst();
            }
            block_offset += out_subblock_num_tiles;
        }
        cb_push_back(reblock_cb_id, out_block_w);

        // Untilize
        untilize_init_short(reblock_cb_id);
        cb_wait_front(reblock_cb_id, out_block_w);
        cb_reserve_back(out_cb_id, out_block_w);
        untilize_block(reblock_cb_id, out_block_w, out_cb_id);
        cb_pop_front(reblock_cb_id, out_block_w);
        cb_push_back(out_cb_id, out_block_w);
        untilize_uninit(reblock_cb_id);

        within_block_index += out_subblock_w;
    }
    cb_pop_front(interm_cb_id, num_tiles_in_row_of_subblocks);
}

inline void pack_matmul_subblock(uint32_t cb_id, uint32_t out_subblock_num_tiles) {
    cb_reserve_back(cb_id, out_subblock_num_tiles);
    for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
        pack_tile(i, cb_id);
    }
    cb_push_back(cb_id, out_subblock_num_tiles);
}

namespace NAMESPACE {
void MAIN {

    uint32_t in0_block_w = get_compile_time_arg_val(0); // inner block size in tiles
    uint32_t in0_num_subblocks = get_compile_time_arg_val(1); // outer row block size (in inner row blocks)
    uint32_t in0_block_num_tiles =  get_compile_time_arg_val(2); // out_subblock_h*in0_block_w*in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);  // out_subblock_h*in0_block_w
    uint32_t in0_subblock_h = get_compile_time_arg_val(4);
    uint32_t in1_num_subblocks = get_compile_time_arg_val(5); // outer column block size (in inner column blocks)
    uint32_t in1_block_num_tiles = get_compile_time_arg_val(6); //out_subblock_w*in0_block_w* in1_num_subblocks;
    uint32_t in1_per_core_w = get_compile_time_arg_val(7); // out_subblock_w*in1_num_subblocks
    // If I don't make this volatile, causes code size for TRISC2 to be too large if num_blocks > 1
    volatile uint32_t num_blocks_in0_h = get_compile_time_arg_val(8);  // outer inner dim (in inner dim blocks)
    volatile uint32_t num_blocks_in0_w = get_compile_time_arg_val(9);  // outer inner dim (in inner dim blocks)
    volatile uint32_t num_blocks_in1_w = get_compile_time_arg_val(10);

    uint32_t out_subblock_h = get_compile_time_arg_val(11); // inner row block size in tiles
    uint32_t out_subblock_w = get_compile_time_arg_val(12); // inner column block size in tiles
    uint32_t out_subblock_num_tiles = get_compile_time_arg_val(13); // out_subblock_h * out_subblock_w;

    uint32_t out_block_w = in1_per_core_w;

    // If true, this assumes data coming in RM
    bool tilize_in = get_compile_time_arg_val(13);

    // If true, this assumes consumer wants data RM
    bool untilize_out = get_compile_time_arg_val(14);

    bool spill = num_blocks_in0_w > 1;

    bool enable_reload = false;

    // CB mapping of in0, union of all possible variants (with
    // and without fusing combinations of tilize/untilize)
    // in0:
    //   input 0
    // in1:
    //   input 1
    // interm0:
    //   If under tilized mode, this is CB in which we write the tilized
    //   input 0
    // interm1:
    //   intermediate CB we write to so that we store partial matmul results
    // interm2:
    //   if under untilize mode, this is the CB we write to so that we store
    //   the final matmul result
    // interm3:
    //   if under untilize mode, this is the CB we write to so that we can
    //   reblock the output
    uint32_t in0_cb                                   = tt::CB::c_in0;
    uint32_t tilize_mode_tilized_in0_cb               = tt::CB::c_intermed0;
    uint32_t matmul_partials_cb                       = tt::CB::c_intermed1;
    uint32_t untilize_mode_final_matmul_partials_cb   = tt::CB::c_intermed2;
    uint32_t untilize_mode_reblock_cb                 = tt::CB::c_intermed3;
    uint32_t out0_cb                                  = tt::CB::c_out0;
    mm_init();
    for(uint32_t block_in0_h = 0; block_in0_h < num_blocks_in0_h; block_in0_h++) {
        for(uint32_t block_in1_w = 0; block_in1_w < num_blocks_in1_w; block_in1_w++) {
            enable_reload = false;
            //DPRINT << 'B' << ENDL();
            for(uint32_t block_in0_w = 0; block_in0_w < num_blocks_in0_w; block_in0_w++)
            {

                bool last_out = block_in0_w == (num_blocks_in0_w-1);
                if  (tilize_in) {
                    tilize_activation(in0_cb, in0_subblock_h, in0_block_w, in0_num_subblocks, tilize_mode_tilized_in0_cb);
                    mm_init_short();
                    cb_wait_front(tilize_mode_tilized_in0_cb, in0_block_num_tiles);

                } else {
                    cb_wait_front(in0_cb, in0_block_num_tiles);
                }

                cb_wait_front(tt::CB::c_in1, in1_block_num_tiles);

                int in0_index_subblock_offset = 0;
                for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                    int in1_index_subblock_offset = 0;
                    for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {

                        acquire_dst();

                        if (enable_reload) {
                            copy_tile_to_dst_init_short();
                            cb_wait_front(matmul_partials_cb, out_subblock_num_tiles);
                            for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                                copy_tile(matmul_partials_cb, i, i);
                            }
                            cb_pop_front(matmul_partials_cb, out_subblock_num_tiles);
                            mm_init_short();
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
                                    if  (tilize_in) {
                                        matmul_tiles(tilize_mode_tilized_in0_cb, tt::CB::c_in1, in0_index, in1_index, dst_index, false /* transpose */);
                                    } else {
                                        matmul_tiles(in0_cb, tt::CB::c_in1, in0_index, in1_index, dst_index, false /* transpose */);
                                    }
                                    in1_index_inner_dim_offset += in1_per_core_w;
                                }
                                dst_index++;
                            }
                            in0_index_h_offset += in0_block_w;
                        }
                        if (last_out) {
                            if  (not untilize_out) {
                                pack_matmul_subblock(out0_cb, out_subblock_num_tiles);
                            } else {
                                pack_matmul_subblock(untilize_mode_final_matmul_partials_cb, out_subblock_num_tiles);
                            }
                        } else {
                            pack_matmul_subblock(matmul_partials_cb, out_subblock_num_tiles);
                        }
                        release_dst();

                        in1_index_subblock_offset += out_subblock_w;
                    }

                    if (untilize_out) {
                        if (last_out) {
                            reblock_and_untilize(
                                in1_num_subblocks,
                                out_subblock_num_tiles,
                                out_subblock_h,
                                out_subblock_w,
                                out_block_w,
                                untilize_mode_final_matmul_partials_cb,
                                untilize_mode_reblock_cb,
                                out0_cb
                            );
                            mm_init_short();
                        }
                    }


                    in0_index_subblock_offset += in0_subblock_num_tiles;
                }

                if  (spill) enable_reload = true;

                if  (tilize_in) {
                    cb_pop_front(tilize_mode_tilized_in0_cb, in0_block_num_tiles);
                } else {
                    cb_pop_front(in0_cb, in0_block_num_tiles);
                }
                cb_pop_front(tt::CB::c_in1, in1_block_num_tiles);
            }
        }

    }


}
}
