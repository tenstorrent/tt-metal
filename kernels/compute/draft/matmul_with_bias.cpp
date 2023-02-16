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

void hlk_main(tt_core *core_ptr, const hlk_args_t *args)
{
    hlk_mm_tile_init(core_ptr, false);
    hlk_acquire_dst(core_ptr, DstMode::Full);

    for(int b=0;b<args->block_cnt;++b)
    {
        hlk_wait_tiles(core_ptr, HlkOperand::in0, args->in0_block_tile_cnt);
        hlk_wait_tiles(core_ptr, HlkOperand::in1, args->in1_block_tile_cnt);
        int dst_tile_index = 0;
        int in0_block_tile_index = 0;
        for(int r=0;r<args->dst_tile_rows;++r)
        {
            for(int c=0;c<args->dst_tile_cols;++c)
            {
                int in1_block_tile_index = 0;
                for(int i=0;i<args->block_tile_dim;++i)
                {
                    hlk_mm_tile(core_ptr, HlkOperand::in0, HlkOperand::in1, in0_block_tile_index+i, in1_block_tile_index+c, dst_tile_index,false);
                    in1_block_tile_index += args->dst_tile_cols;
                }
                dst_tile_index++;
            }
            in0_block_tile_index += args->block_tile_dim;
        }
        hlk_pop_tiles(core_ptr, HlkOperand::in0, args->in0_block_tile_cnt);
        hlk_pop_tiles(core_ptr, HlkOperand::in1, args->in1_block_tile_cnt);
    }

    // Pack out
    hlk_wait_for_free_tiles(core_ptr, HlkOperand::intermed0, args->out_block_tile_cnt); 
    for(int i=0 ; i<args->out_block_tile_cnt;++i)
    {
        hlk_pack_tile_to_stream(core_ptr, i, HlkOperand::intermed0);
    }

    hlk_push_tiles(core_ptr, HlkOperand::intermed0, args->out_block_tile_cnt); 

    hlk_release_dst(core_ptr, DstMode::Full);

    hlk_add_tile_bcast_init(core_ptr);
    hlk_acquire_dst(core_ptr, DstMode::Full);
    hlk_wait_tiles(core_ptr, HlkOperand::intermed0, args->out_block_tile_cnt);

    hlk_wait_tiles(core_ptr, HlkOperand::in2, args->dst_tile_cols);
    int dst_tile_index = 0;
    for(int r=0;r<args->dst_tile_rows;++r)
    {
        for(int c=0;c<args->dst_tile_cols;++c)
        {
            hlk_add_tile_bcast(core_ptr, (int)Dim::R, HlkOperand::intermed0, HlkOperand::in2, dst_tile_index, c, dst_tile_index);
            dst_tile_index++;
        }
    }
    hlk_pop_tiles(core_ptr, HlkOperand::in2, args->dst_tile_cols);

    hlk_pop_tiles(core_ptr, HlkOperand::intermed0, args->out_block_tile_cnt);

    hlk_wait_for_free_tiles(core_ptr, HlkOperand::out0, args->out_block_tile_cnt); 
    for(int i = 0; i < args->out_block_tile_cnt ; ++i) {
        hlk_pack_tile_to_stream(core_ptr, i, HlkOperand::out0);
    }
    hlk_push_tiles(core_ptr, HlkOperand::out0, args->out_block_tile_cnt);
    
    hlk_release_dst(core_ptr, DstMode::Full);
}
