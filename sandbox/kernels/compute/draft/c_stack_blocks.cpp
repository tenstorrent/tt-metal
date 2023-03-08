#include <cstdint>

#include "compute_hlk_api.h"

struct hlk_args_t {
    // stack a number of inputs block into an output block
    std::int32_t slice_factor;

    // input tensor dims
    std::int32_t num_input_blocks;
    std::int32_t num_tiles_per_input_block;
    std::int32_t input_block_shape_r;
    std::int32_t input_block_shape_c;

    // output tensor dims
    std::int32_t num_output_blocks; // can be interpreted as "num blocks", "num tensors", "z-dim", or "batch" loop
    std::int32_t num_tiles_per_output_block;
};

// do this for "num blocks" (per_core_num_blocks)
//    In0: an RM block with dims (per_core_in_r_tile, input_block_shape_c)
//    Out0: create "slice_factor" output blocks, each with (per_core_in_r_tile, per_block_out_c_tiles)
void hlk_main(tt_core *core_ptr, const hlk_args_t *args) {

    for (int output_block_index = 0; output_block_index < args->num_output_blocks; output_block_index++) {

        hlk_wait_tiles(core_ptr, HlkOperand::in0, args->num_tiles_per_output_block);

        int r_offset = 0;
        for(int block_index_r = 0; block_index_r < args->input_block_shape_r ; ++block_index_r) {
            int slice_offset = 0;
            for(int slice_index = 0; slice_index < args->slice_factor ; ++slice_index) {
                for(int block_index_c = 0; block_index_c < args->input_block_shape_c ; ++block_index_c) {
                    int index = slice_offset + r_offset + block_index_c;
                    hlk_acquire_dst(core_ptr, DstMode::Half);
                    hlk_copy_tile_to_dst(core_ptr, HlkOperand::in0, index, 0);

                    hlk_wait_for_free_tiles(core_ptr, HlkOperand::out0, 1);
                    hlk_pack_tile_to_stream(core_ptr, 0, HlkOperand::out0);
                    hlk_push_tiles(core_ptr, HlkOperand::out0, 1);

                    hlk_release_dst(core_ptr, DstMode::Half);
                }
                slice_offset += args->num_tiles_per_input_block;
            }
            r_offset += args->input_block_shape_c;
        }
        hlk_pop_tiles(core_ptr, HlkOperand::in0, args->num_tiles_per_output_block);

    }
}
