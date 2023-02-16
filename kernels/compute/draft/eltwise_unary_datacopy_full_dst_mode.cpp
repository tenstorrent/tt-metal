#include <cstdint>

#include "compute_hlk_api.h"

struct hlk_args_t {
    std::int32_t per_core_tile_cnt;
    std::int32_t per_core_block_cnt;
    std::int32_t per_core_block_dim;
};
  
void hlk_main(tt_core *core_ptr, const hlk_args_t *args) {
    hlk_acquire_dst(core_ptr, DstMode::Full);

    int dst_tile_index = 0;
    for(int i = 0; i < args->per_core_block_cnt ; ++i) {
        hlk_wait_tiles(core_ptr, HlkOperand::in0, args->per_core_block_dim);

        for(int j = 0; j < args->per_core_block_dim ; ++j) {
            hlk_copy_tile_to_dst(core_ptr, HlkOperand::in0, j, dst_tile_index);
            dst_tile_index++;
        }

        hlk_pop_tiles(core_ptr, HlkOperand::in0, args->per_core_block_dim);
    }

    hlk_wait_for_free_tiles(core_ptr, HlkOperand::out0, args->per_core_tile_cnt); 
    for(int i = 0; i < args->per_core_tile_cnt ; ++i) {
        hlk_pack_tile_to_stream(core_ptr, i, HlkOperand::out0);
    }
    hlk_push_tiles(core_ptr, HlkOperand::out0, args->per_core_tile_cnt); 

    hlk_release_dst(core_ptr, DstMode::Full);

}
