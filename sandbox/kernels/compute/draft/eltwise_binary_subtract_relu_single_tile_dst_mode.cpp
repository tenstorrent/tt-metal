#include <cstdint>

#include "compute_hlk_api.h"

struct hlk_args_t {
    std::int32_t per_core_block_cnt;
    std::int32_t per_core_block_dim;
};

void hlk_main(tt_core *core_ptr, const hlk_args_t *args) {

    for(int block = 0; block < args->per_core_block_cnt; ++block) {
        hlk_wait_for_free_tiles(core_ptr, HlkOperand::out0, args->per_core_block_dim);
        for(int t = 0; t < args->per_core_block_dim; ++t)
        {
            hlk_acquire_dst(core_ptr, DstMode::Half);

            // Wait for tiles on the input
            hlk_wait_tiles(core_ptr, HlkOperand::in0, 1);
            hlk_wait_tiles(core_ptr, HlkOperand::in1, 1);
            // Wait for space in output

            // Subtract and pack
            hlk_subtract_tile(core_ptr, HlkOperand::in0, HlkOperand::in1, 0, 0, 0);
            hlk_pack_relu_tile_to_stream(core_ptr, 0, HlkOperand::out0);

            // Pop input and push to output
            hlk_pop_tiles(core_ptr, HlkOperand::in0, 1);
            hlk_pop_tiles(core_ptr, HlkOperand::in1, 1);

            hlk_release_dst(core_ptr, DstMode::Half);
        }
        hlk_push_tiles(core_ptr, HlkOperand::out0, args->per_core_block_dim);
    }
}
