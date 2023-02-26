#include <cstdint>

#include "compute_hlk_api.h"

struct hlk_args_t {
    std::int32_t per_core_num_blocks;
    std::int32_t per_block_r_tiles;
    std::int32_t per_block_c_tiles;
};

void hlk_main(tt_core *core_ptr, const hlk_args_t *args) {
    hlk_wait_tiles(core_ptr, HlkOperand::in1, 1);
    for (int block_index = 0; block_index < args->per_core_num_blocks; block_index++) {
        for(int r = 0; r < args->per_block_r_tiles ; ++r) {
            hlk_wait_for_free_tiles(core_ptr, HlkOperand::out0, args->per_block_c_tiles);
            for (int c = 0; c < args->per_block_c_tiles; c++) {
                hlk_acquire_dst(core_ptr, DstMode::Half);
                hlk_wait_tiles(core_ptr, HlkOperand::in0, 1);

                hlk_multiply_tile_bcast(core_ptr, (int)Dim::RC, HlkOperand::in0, HlkOperand::in1, 0, 0, 0);
                hlk_pop_tiles(core_ptr, HlkOperand::in0, 1);

                hlk_pack_tile_to_stream(core_ptr, 0, HlkOperand::out0);

                hlk_release_dst(core_ptr, DstMode::Half);
            }
            hlk_push_tiles(core_ptr, HlkOperand::out0, args->per_block_c_tiles);
        }
    }
    hlk_pop_tiles(core_ptr, HlkOperand::in1, 1);
}
