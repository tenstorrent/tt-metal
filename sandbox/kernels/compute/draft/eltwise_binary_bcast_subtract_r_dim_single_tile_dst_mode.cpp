#include <cstdint>

#include "compute_hlk_api.h"

struct hlk_args_t {
    std::int32_t tensor_z;
    std::int32_t num_blocks_r;
    std::int32_t num_blocks_c;
    std::int32_t block_shape_r;
    std::int32_t block_shape_c;
};

void hlk_main(tt_core *core_ptr, const hlk_args_t *args) {
    for (int z = 0; z < args->tensor_z; ++z) {
        for (int block_r = 0; block_r < args->num_blocks_r; block_r++) {
            for (int block_c = 0; block_c < args->num_blocks_c; block_c++) {
                if (block_r == 0) {
                    hlk_wait_tiles(core_ptr, HlkOperand::in1, args->block_shape_c);
                }

                for (int i = 0; i < args->block_shape_r; ++i) {
                    hlk_wait_tiles(core_ptr, HlkOperand::in0, args->block_shape_c);
                    hlk_wait_for_free_tiles(core_ptr, HlkOperand::out0, args->block_shape_c);

                    for (int j = 0; j < args->block_shape_c; ++j) {
                        hlk_acquire_dst(core_ptr, DstMode::Half);
                        hlk_subtract_tile_bcast(core_ptr, (int)Dim::R, HlkOperand::in0, HlkOperand::in1, j, j, 0);
                        hlk_pack_tile_to_stream(core_ptr, 0, HlkOperand::out0);
                        hlk_release_dst(core_ptr, DstMode::Half);
                    }
                    hlk_push_tiles(core_ptr, HlkOperand::out0, args->block_shape_c);
                    hlk_pop_tiles(core_ptr, HlkOperand::in0, args->block_shape_c);
                }

                if (block_r == (args->num_blocks_r - 1)) {
                    hlk_pop_tiles(core_ptr, HlkOperand::in1, args->block_shape_c);
                }
            }
        }
    }
}
