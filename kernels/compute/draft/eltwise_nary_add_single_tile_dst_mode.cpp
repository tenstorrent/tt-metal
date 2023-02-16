#include <cstdint>

#include "compute_hlk_api.h"

struct hlk_args_t {
    std::int32_t per_core_tile_cnt;
    std::int32_t input_count;
};
  
void hlk_main(tt_core *core_ptr, const hlk_args_t *args) {

    for(int t = 0; t < args->per_core_tile_cnt; ++t)
    {
        hlk_acquire_dst(core_ptr, DstMode::Half);
        

        // Wait for tiles on the input
        for (int i=0; i < args->input_count; i++) 
        {
            // Accumulate
            hlk_wait_tiles(core_ptr, HlkOperand::in0 + i, 1);
            hlk_add_tile_to_dst(core_ptr, HlkOperand::in0 + i, 0, 0, i==0);
            hlk_pop_tiles(core_ptr, HlkOperand::in0 + i, 1);
        }

        // Wait for space in output
        hlk_wait_for_free_tiles(core_ptr, HlkOperand::out0, 1);
        hlk_pack_tile_to_stream(core_ptr, 0, HlkOperand::out0);
        hlk_push_tiles(core_ptr, HlkOperand::out0, 1);

        hlk_release_dst(core_ptr, DstMode::Half);
    }
}

