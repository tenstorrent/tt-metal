#include <cstdint>

#include "compute_hlk_api.h"

struct hlk_args_t {
    int block_tile_dim;
    int dst_tile_rows;
    int dst_tile_cols;
    int block_cnt;
    int in0_block_tile_cnt;
    int in1_block_tile_cnt;
    int out_block_tile_cnt;
};


void compute_main(const hlk_args_t *args) {
    acquire_dst(DstMode::Full);

    for(int b=0;b<args->block_cnt;++b)
    {
        cb_wait_front(CB::c_in0, args->in0_block_tile_cnt);
        cb_wait_front(CB::c_in1, args->in1_block_tile_cnt);
        int dst_tile_index = 0;
        int in0_block_tile_index = 0;
        for(int r=0;r<args->dst_tile_rows;++r)
        {
            for(int c=0;c<args->dst_tile_cols;++c)
            {
                int in1_block_tile_index = 0;
                for(int i=0;i<args->block_tile_dim;++i)
                {
                    matmul_tiles(CB::c_in0, CB::c_in1, in0_block_tile_index+i, in1_block_tile_index+c, dst_tile_index, false);
                    in1_block_tile_index += args->dst_tile_cols;
                }
                dst_tile_index++;
            }
            in0_block_tile_index += args->block_tile_dim;
        }
        cb_pop_front(CB::c_in0, args->in0_block_tile_cnt);
        cb_pop_front(CB::c_in1, args->in1_block_tile_cnt);
    }

    // Pack out
    cb_reserve_back(CB::c_out0, args->out_block_tile_cnt);
    for(int i=0 ; i<args->out_block_tile_cnt;++i)
    {
        pack_tile(i, CB::c_out0);
    }

    cb_push_back(CB::c_out0, args->out_block_tile_cnt);

    release_dst(DstMode::Full);
}
