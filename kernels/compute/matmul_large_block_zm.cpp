#include <cstdint>

#include "compute_hlk_api.h"

struct hlk_args_t {
    int in0_block_w;
    int in0_num_subblocks;
    int in0_block_num_tiles;
    int in0_subblock_num_tiles;
    int in1_num_subblocks;
    int in1_block_num_tiles;
    int in1_per_core_w;
    int num_blocks;
    int out_subblock_h;
    int out_subblock_w;
    int out_subblock_num_tiles;
};

void compute_main(const hlk_args_t *args) {

    bool spill = args->num_blocks > 1;
    int out_subblock_h = args->out_subblock_h; // inner row block size in tiles
    int out_subblock_w = args->out_subblock_w; // inner column block size in tiles
    int in0_block_w = args->in0_block_w; // inner block size in tiles
    int in0_num_subblocks = args->in0_num_subblocks; // outer row block size (in inner row blocks)
    int in1_num_subblocks = args->in1_num_subblocks; // outer column block size (in inner column blocks)
    int num_blocks = args->num_blocks; // outer inner dim (in inner dim blocks)
    int in0_block_num_tiles = args->in0_block_num_tiles; //out_subblock_h*in0_block_w*in0_num_subblocks;
    int in1_block_num_tiles = args->in1_block_num_tiles; //out_subblock_w*in0_block_w*in1_num_subblocks;
    int out_subblock_num_tiles = args->out_subblock_num_tiles; //out_subblock_h * out_subblock_w;
    int in0_subblock_num_tiles = args->in0_subblock_num_tiles; // out_subblock_h*in0_block_w
    int in1_per_core_w = args->in1_per_core_w; // out_subblock_w*in1_num_subblocks

    matmul_tile_init_once(0);
    bool enable_reload = false;

    for(int block = 0; block < num_blocks; block++)
    {
        bool last_out = block == (num_blocks-1);

        cb_wait_front(CB::c_in0, in0_block_num_tiles);
        cb_wait_front(CB::c_in1, in1_block_num_tiles);
        int in0_index_subblock_offset = 0;
        for (int in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
            int in1_index_subblock_offset = 0;
            for (int in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {

                acquire_dst(DstMode::Half);

                if (enable_reload) {
                    hlk_copy_tile_to_dst_init_short(nullptr);
                    cb_wait_front(CB::c_intermed0, out_subblock_num_tiles);
                    for (int i = 0; i < out_subblock_num_tiles; i++) {
                        copy_tile(CB::c_intermed0, i, i);
                    }
                    cb_pop_front(CB::c_intermed0, out_subblock_num_tiles);
                    matmul_tile_init_short(0);
                }

                // Compute output sub-block from in0_subblock x in1_subblock
                int dst_index = 0;
                int in0_index_h_offset = 0;
                for (int h = 0; h < out_subblock_h; h++) {
                    for (int w = 0; w < out_subblock_w; w++) {
                        int in1_index_inner_dim_offset = 0;
                        for (int inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                            int in0_index = in0_index_subblock_offset + in0_index_h_offset + inner_dim;
                            int in1_index = in1_index_subblock_offset + in1_index_inner_dim_offset + w;
                            matmul_tiles(CB::c_in0, CB::c_in1, in0_index, in1_index, dst_index, false /* transpose */);
                            in1_index_inner_dim_offset += in1_per_core_w;
                        }
                        dst_index++;
                    }
                    in0_index_h_offset += in0_block_w;
                }

                if (last_out) {
                    // Pack out to output buffer
                    cb_reserve_back(CB::c_out0, out_subblock_num_tiles);
                    for (int i = 0; i < out_subblock_num_tiles; i++) {
                        pack_tile(i, CB::c_out0);
                    }
                    cb_push_back(CB::c_out0, out_subblock_num_tiles);
                } else {
                    // Move partial result to interm buffer
                    cb_reserve_back(CB::c_intermed0, out_subblock_num_tiles);
                    for (int i = 0; i < out_subblock_num_tiles; i++) {
                        pack_tile(i, CB::c_intermed0);
                    }
                    cb_push_back(CB::c_intermed0, out_subblock_num_tiles);
                }

                release_dst(DstMode::Half);
                in1_index_subblock_offset += out_subblock_w;
            }
            in0_index_subblock_offset += in0_subblock_num_tiles;
        }

        if (spill) enable_reload = true;

        cb_pop_front(CB::c_in0, in0_block_num_tiles);
        cb_pop_front(CB::c_in1, in1_block_num_tiles);

    }

}
