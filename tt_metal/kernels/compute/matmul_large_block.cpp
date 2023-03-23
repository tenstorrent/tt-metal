#include <cstdint>

#include "llk_3c.h"

inline void tilize_activation(uint32_t in0_subblock_h, uint32_t in0_block_w, uint32_t in0_num_subblocks, uint32_t out_cb)
{
    tilize_init_short(CB::c_in0, in0_block_w);

    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
        for (uint32_t h = 0; h < in0_subblock_h; h++) {
            cb_wait_front(CB::c_in0, in0_block_w);
            cb_reserve_back(out_cb, in0_block_w);
            tilize_block(CB::c_in0, in0_block_w, out_cb);
            cb_push_back(out_cb, in0_block_w);
            cb_pop_front(CB::c_in0, in0_block_w);
        }
    }

    tilize_uninit();
}

inline void reblock_and_untilize(
    uint32_t num_out_subblocks_in_col,
    uint32_t out_subblock_num_tiles,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t out_block_w,
    uint32_t interm_cb_id,
    uint32_t reblock_cb_id)
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
                acquire_dst(DstMode::Half);
                copy_tile(interm_cb_id, tile_index, 0);
                pack_tile(0, reblock_cb_id);
                release_dst(DstMode::Half);
            }
            block_offset += out_subblock_num_tiles;
        }
        cb_push_back(reblock_cb_id, out_block_w);

        // Untilize
        untilize_init_short(reblock_cb_id);
        cb_wait_front(reblock_cb_id, out_block_w);
        cb_reserve_back(CB::c_out0, out_block_w);
        untilize_block(reblock_cb_id, out_block_w, CB::c_out0);
        cb_pop_front(reblock_cb_id, out_block_w);
        cb_push_back(CB::c_out0, out_block_w);
        untilize_uninit(reblock_cb_id);

        within_block_index += out_subblock_w;
    }
    cb_pop_front(interm_cb_id, num_tiles_in_row_of_subblocks);
}

namespace NAMESPACE {
void MAIN {

    uint32_t in0_block_w = get_compile_time_arg_val(0); // inner block size in tiles
    uint32_t in0_num_subblocks = get_compile_time_arg_val(1); // outer row block size (in inner row blocks)
    uint32_t in0_block_num_tiles = get_compile_time_arg_val(2); // out_subblock_h*in0_block_w*in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);  // out_subblock_h*in0_block_w
     uint32_t in0_subblock_h = get_compile_time_arg_val(4);
    uint32_t in1_num_subblocks = get_compile_time_arg_val(5); // outer column block size (in inner column blocks)
    uint32_t in1_block_num_tiles = get_compile_time_arg_val(6); //out_subblock_w*in0_block_w* in1_num_subblocks;
    uint32_t in1_per_core_w = get_compile_time_arg_val(7); // out_subblock_w*in1_num_subblocks
    constexpr uint32_t num_blocks = get_compile_time_arg_val(8);  // outer inner dim (in inner dim blocks)
    uint32_t out_subblock_h = get_compile_time_arg_val(9); // inner row block size in tiles
    uint32_t out_subblock_w = get_compile_time_arg_val(10); // inner column block size in tiles
    uint32_t out_subblock_num_tiles = get_compile_time_arg_val(11); // out_subblock_h * out_subblock_w;

    uint32_t out_block_w = in1_per_core_w;

    // If true, this assumes data coming in RM
    constexpr bool tilize_in = get_compile_time_arg_val(12);

    // If true, this assumes consumer wants data RM
    constexpr bool untilize_out = get_compile_time_arg_val(13);

    constexpr bool spill = num_blocks > 1;

    bool enable_reload = false;

    uint32_t matmul_act_cb_id = 0;
    uint32_t matmul_out_intermediate_cb_id = 24;
    if constexpr (tilize_in) {
        // If we tilize, matmul doesn't consume original input,
        // it consumes what is produced by tilize
        matmul_act_cb_id = 24;
        matmul_out_intermediate_cb_id = 25; // Given 24 is no longer available, we use 25 instead
    }

    uint32_t reblock_cb_id = 26; // Only used if untilize is required

    mm_init();
    for(uint32_t block = 0; block < num_blocks; block++)
    {
        bool last_out = block == (num_blocks-1);

        if constexpr (tilize_in) {
            tilize_activation(in0_subblock_h, in0_block_w, in0_num_subblocks, matmul_act_cb_id);
        }

        cb_wait_front(matmul_act_cb_id, in0_block_num_tiles);
        cb_wait_front(CB::c_in1, in1_block_num_tiles);
        int in0_index_subblock_offset = 0;
        for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
            int in1_index_subblock_offset = 0;
            for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {

                acquire_dst(DstMode::Half);

                if (enable_reload) {
                    copy_tile_to_dst_init_short();
                    cb_wait_front(matmul_out_intermediate_cb_id, out_subblock_num_tiles);
                    for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                        copy_tile(matmul_out_intermediate_cb_id, i, i);
                    }
                    cb_pop_front(matmul_out_intermediate_cb_id, out_subblock_num_tiles);
                }
                mm_init_short();

                // Compute output sub-block from in0_subblock x in1_subblock
                int dst_index = 0;
                int in0_index_h_offset = 0;
                for (uint32_t h = 0; h < out_subblock_h; h++) {
                    for (uint32_t w = 0; w < out_subblock_w; w++) {
                        int in1_index_inner_dim_offset = 0;
                        for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                            int in0_index = in0_index_subblock_offset + in0_index_h_offset + inner_dim;
                            int in1_index = in1_index_subblock_offset + in1_index_inner_dim_offset + w;
                            matmul_tiles(matmul_act_cb_id, CB::c_in1, in0_index, in1_index, dst_index, false /* transpose */);
                            in1_index_inner_dim_offset += in1_per_core_w;
                        }
                        dst_index++;
                    }
                    in0_index_h_offset += in0_block_w;
                }

                if constexpr (not untilize_out) {
                    if (last_out) {
                        // Pack out to output buffer
                        cb_reserve_back(CB::c_out0, out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            pack_tile(i, CB::c_out0);
                        }
                        cb_push_back(CB::c_out0, out_subblock_num_tiles);
                    } else {
                        // Move partial result to interm buffer
                        cb_reserve_back(matmul_out_intermediate_cb_id, out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            pack_tile(i, matmul_out_intermediate_cb_id);
                        }
                        cb_push_back(matmul_out_intermediate_cb_id, out_subblock_num_tiles);
                    }
                } else {
                    // Move partial result to interm buffer
                    cb_reserve_back(matmul_out_intermediate_cb_id, out_subblock_num_tiles);
                    for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                        pack_tile(i, matmul_out_intermediate_cb_id);
                    }
                    cb_push_back(matmul_out_intermediate_cb_id, out_subblock_num_tiles);
                }

                release_dst(DstMode::Half);

                in1_index_subblock_offset += out_subblock_w;
            }

            if constexpr (untilize_out) {
                if (last_out) {
                    reblock_and_untilize(
                        in1_num_subblocks,
                        out_subblock_num_tiles,
                        out_subblock_h,
                        out_subblock_w,
                        out_block_w,
                        matmul_out_intermediate_cb_id,
                        reblock_cb_id
                    );
                }
            }

            in0_index_subblock_offset += in0_subblock_num_tiles;
        }

        if constexpr (spill) enable_reload = true;

        cb_pop_front(matmul_act_cb_id, in0_block_num_tiles);
        cb_pop_front(CB::c_in1, in1_block_num_tiles);

    }

}
}
