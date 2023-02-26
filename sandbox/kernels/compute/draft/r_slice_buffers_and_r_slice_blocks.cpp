#include <cstdint>

#include "compute_hlk_api.h"

struct hlk_args_t {
    std::int32_t num_output_buffers;       // the number of outputs to scatter to

    // input tensor dims, from each input we get (per_core_num_in_blocks_tiles * per_core_in_block_tiles) tiles
    std::int32_t num_input_blocks;
    std::int32_t input_block_shape_c;
    std::int32_t output_block_shape_r;
};

void hlk_main(tt_core *core_ptr, const hlk_args_t *args) {

    for (int block_index = 0; block_index < args->num_input_blocks; block_index++) {
        for(int output_index = 0; output_index < args->num_output_buffers ; output_index++) {
            for (int tile_index_r = 0; tile_index_r < args->output_block_shape_r; tile_index_r++) {
                hlk_wait_for_free_tiles(core_ptr, HlkOperand::out0 + output_index, args->input_block_shape_c);
                for(int tile_index_c = 0; tile_index_c < args->input_block_shape_c ; tile_index_c++) {
                    hlk_acquire_dst(core_ptr, DstMode::Half);

                    // wait and pop tile-by-tile -- such that we can use minimally sized buffers for each of the inputs
                    hlk_wait_tiles(core_ptr, HlkOperand::in0, 1);

                    hlk_copy_tile_to_dst(core_ptr, HlkOperand::in0, 0, 0);
                    hlk_pack_tile_to_stream(core_ptr, 0, HlkOperand::out0 + output_index);

                    hlk_pop_tiles(core_ptr, HlkOperand::in0 , 1);

                    hlk_release_dst(core_ptr, DstMode::Half);
                }
                hlk_push_tiles(core_ptr, HlkOperand::out0 + output_index, args->input_block_shape_c);
            }
        }
    }
}
