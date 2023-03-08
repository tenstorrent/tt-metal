#include <cstdint>

#include "compute_hlk_api.h"

struct hlk_args_t {
    int batch_size;

    // per-batch params
    int per_block_m_tiles;
    int per_block_k_tiles;
    int per_block_n_tiles;
    int per_batch_input_block_cnt;
    int in0_block_tile_cnt;
    int in1_block_tile_cnt;
    int out_block_tile_cnt;
};

void hlk_main(tt_core *core_ptr, const hlk_args_t *args)
{
    for (int b = 0; b < args->batch_size; b++) {

        hlk_acquire_dst(core_ptr, DstMode::Full);

        for(int in_block_idx=0; in_block_idx < args->per_batch_input_block_cnt; ++in_block_idx)
        {
            hlk_wait_tiles(core_ptr, CB::c_in0, args->in0_block_tile_cnt);
            hlk_wait_tiles(core_ptr, CB::c_in1, args->in1_block_tile_cnt);

            int dst_tile_index = 0;
            int in0_block_tile_index = 0;

            for(int r=0;r<args->per_block_m_tiles;++r)
            {
                for(int c=0;c<args->per_block_n_tiles;++c)
                {
                    int in1_block_tile_index = 0;
                    for(int i=0;i<args->per_block_k_tiles;++i)
                    {
                        hlk_mm_tile(core_ptr, CB::c_in0, CB::c_in1, in0_block_tile_index+i, in1_block_tile_index+c, dst_tile_index,false);
                        in1_block_tile_index += args->per_block_n_tiles;
                    }
                    dst_tile_index++;
                }
                in0_block_tile_index += args->per_block_k_tiles;
            }
            hlk_pop_tiles(core_ptr, CB::c_in0, args->in0_block_tile_cnt);
            hlk_pop_tiles(core_ptr, CB::c_in1, args->in1_block_tile_cnt);
        }

        // Pack out
        hlk_wait_for_free_tiles(core_ptr, CB::c_out0, args->out_block_tile_cnt);
        for(int i=0 ; i<args->out_block_tile_cnt;++i)
        {
            hlk_pack_tile_to_stream(core_ptr, i, CB::c_out0);
        }

        hlk_push_tiles(core_ptr, CB::c_out0, args->out_block_tile_cnt);

        hlk_release_dst(core_ptr, DstMode::Full);
    }
}
