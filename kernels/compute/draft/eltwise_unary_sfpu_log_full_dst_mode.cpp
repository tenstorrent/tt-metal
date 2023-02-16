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
    for(int block_index = 0; block_index < args->per_core_block_cnt ; ++block_index) {
        hlk_wait_tiles(core_ptr, HlkOperand::in0, args->per_core_block_dim);

        for(int in_tile_index = 0; in_tile_index < args->per_core_block_dim ; ++in_tile_index) {
            hlk_copy_tile_to_dst(core_ptr, HlkOperand::in0, in_tile_index, dst_tile_index);
            dst_tile_index++;
        }

        hlk_pop_tiles(core_ptr, HlkOperand::in0, args->per_core_block_dim);
    }

    for(int tile_index = 0; tile_index < args->per_core_tile_cnt ; ++tile_index) {
        hlk_sfpu_log(core_ptr, tile_index);
    }

    hlk_wait_for_free_tiles(core_ptr, HlkOperand::out0, args->per_core_tile_cnt);
    for(int tile_index = 0; tile_index < args->per_core_tile_cnt ; ++tile_index) {
        hlk_pack_tile_to_stream(core_ptr, tile_index, HlkOperand::out0);
    }
    hlk_push_tiles(core_ptr, HlkOperand::out0, args->per_core_tile_cnt);

    hlk_release_dst(core_ptr, DstMode::Full);

}
