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
    int with_bias;
};



void compute_main(const hlk_args_t *args) {
    acquire_dst(DstMode::Full);

    hlk_mm_tile_init_once(nullptr, false);
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


    // add bias in2 to intermed0 and load to dst
    if (args->with_bias) {
        // Pack out
        cb_reserve_back(CB::c_intermed0, args->out_block_tile_cnt);
        for(int i=0 ; i<args->out_block_tile_cnt;++i)
        {
            pack_tile(i, CB::c_intermed0);
        }
        cb_push_back(CB::c_intermed0, args->out_block_tile_cnt);
        release_dst(DstMode::Full);

        acquire_dst(DstMode::Full);
        hlk_add_tile_bcast_init_short(nullptr);
        cb_wait_front(CB::c_intermed0, args->out_block_tile_cnt);
        cb_wait_front(CB::c_in2, args->dst_tile_cols);
        int dst_tile_index = 0;
        for(int r=0;r<args->dst_tile_rows;++r)
        {
            for(int c=0;c<args->dst_tile_cols;++c)
            {
                hlk_add_tile_bcast(core_ptr, (int)Dim::R, HlkOperand::intermed0, HlkOperand::in2, dst_tile_index, c, dst_tile_index);
                dst_tile_index++;
            }
        }
        cb_pop_front(CB::c_in2, args->dst_tile_cols);
    }

    // Pack to c_out0
    cb_reserve_back(CB::c_out0, args->out_block_tile_cnt);
    for(int i=0;i<args->out_block_tile_cnt;++i)
    {
        pack_tile(i, CB::c_out0);
    }

    cb_push_back(CB::c_out0, args->out_block_tile_cnt);
    release_dst(DstMode::Full);
}
