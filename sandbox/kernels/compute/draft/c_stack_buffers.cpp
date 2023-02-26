#include <cstdint>

#include "compute_hlk_api.h"

struct hlk_args_t {
    // the number of inputs to gather
    std::int32_t num_input_buffers;

    // input tensor dims, from each input we get (per_core_num_in_blocks_tiles * per_core_in_block_tiles) tiles
    std::int32_t tensor_z;
    std::int32_t num_blocks_r;
    std::int32_t num_blocks_c;
    std::int32_t block_shape_r;
    std::int32_t block_shape_c;
};

void hlk_main(tt_core *core_ptr, const hlk_args_t *args) {

    for (int z_index = 0; z_index < args->tensor_z; z_index++) {
        for (int block_index_r = 0; block_index_r < args->num_blocks_r; block_index_r++) {
            for (int input_index = 0; input_index < args->num_input_buffers; input_index++) {
                for (int block_index_c = 0; block_index_c < args->num_blocks_c; block_index_c++) {
                    for (int tile_index_r = 0; tile_index_r < args->block_shape_r; tile_index_r++) {
                        for (int tile_index_c = 0; tile_index_c < args->block_shape_c; tile_index_c++) {
                            hlk_acquire_dst(core_ptr, DstMode::Half);

                            // wait and pop tile-by-tile -- such that we can use minimally sized buffers for each of the inputs
                            hlk_wait_tiles(core_ptr, HlkOperand::in0 + input_index, 1);
                            hlk_wait_for_free_tiles(core_ptr, HlkOperand::out0, 1);

                            hlk_copy_tile_to_dst(core_ptr, HlkOperand::in0 + input_index, 0, 0);
                            hlk_pack_tile_to_stream(core_ptr, 0, HlkOperand::out0);

                            hlk_pop_tiles(core_ptr, HlkOperand::in0 + input_index, 1);
                            hlk_push_tiles(core_ptr, HlkOperand::out0, 1);

                            hlk_release_dst(core_ptr, DstMode::Half);
                        }
                    }
                }
            }
        }
    }
}
