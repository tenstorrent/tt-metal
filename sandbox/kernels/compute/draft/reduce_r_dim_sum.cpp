#include <cstdint>

#include "compute_hlk_api.h"

struct hlk_args_t {
    int batch_size;

    // per-batch params
    int per_core_in_r;
    int per_core_in_c;
    int per_core_in_block_cnt;
    int per_core_in_block_size;
    float coefficient;
};

void hlk_main(tt_core *core_ptr, const hlk_args_t *args)
{
    for (int b = 0; b < args->batch_size; b++) {

        hlk_acquire_dst(core_ptr, DstMode::Full);

        // reduce across blocks in the col dim (they all reduce onto per_core_in_r)
        for(int in_block_idx=0; in_block_idx < args->per_core_in_block_cnt; ++in_block_idx)
        {
            hlk_wait_tiles(core_ptr, HlkOperand::in0, args->per_core_in_block_size);

            int input_tile_index = 0;
            for(int r=0;r < args->per_core_in_r; ++r)
            {
                // reduce a row within a block
                for(int c = 0;c < args->per_core_in_c; ++c)
                {
                    int dst_tile_index = c;
                    hlk_reduce_tile(core_ptr, (int)ReduceFunc::Sum, (int)Dim::R, HlkOperand::in0, input_tile_index, dst_tile_index, args->coefficient);
                    input_tile_index++;
                }
            }
            hlk_pop_tiles(core_ptr, HlkOperand::in0, args->per_core_in_block_size);
        }

        // Pack out
        hlk_wait_for_free_tiles(core_ptr, HlkOperand::out0, args->per_core_in_c);
        for(int i=0 ; i<args->per_core_in_c;++i)
        {
            hlk_pack_tile_to_stream(core_ptr, i, HlkOperand::out0);
        }
        hlk_push_tiles(core_ptr, HlkOperand::out0, args->per_core_in_c);

        hlk_release_dst(core_ptr, DstMode::Full);
    }
}
