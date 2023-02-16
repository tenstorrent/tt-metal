#include <cstdint>

#include "compute_hlk_api.h"

struct hlk_args_t {
    std::int32_t per_core_tile_cnt;
    std::int32_t per_block_r_tiles;
    std::int32_t per_block_c_tiles;
};

void hlk_main(tt_core *core_ptr, const hlk_args_t *args) {
    hlk_acquire_dst(core_ptr, DstMode::Full);

    hlk_wait_tiles(core_ptr, HlkOperand::in1, 1);

    int dst_tile_index = 0;
    for(int i = 0; i < args->per_block_r_tiles ; ++i) {
        hlk_wait_tiles(core_ptr, HlkOperand::in0, args->per_block_c_tiles);

        for(int j = 0; j < args->per_block_c_tiles ; ++j) {
            hlk_subtract_tile_bcast(core_ptr, (int)Dim::RC, HlkOperand::in0, HlkOperand::in1, j, 0, dst_tile_index);
            dst_tile_index++;
        }

        hlk_pop_tiles(core_ptr, HlkOperand::in0, args->per_block_c_tiles);
    }
    hlk_pop_tiles(core_ptr, HlkOperand::in1, 1);

    hlk_wait_for_free_tiles(core_ptr, HlkOperand::out0, args->per_core_tile_cnt);
    for(int i = 0; i < args->per_core_tile_cnt ; ++i) {
        hlk_pack_tile_to_stream(core_ptr, i, HlkOperand::out0);
    }
    hlk_push_tiles(core_ptr, HlkOperand::out0, args->per_core_tile_cnt);

    hlk_release_dst(core_ptr, DstMode::Full);

}
