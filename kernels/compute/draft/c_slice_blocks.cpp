#include <cstdint>

#include "compute_hlk_api.h"

struct hlk_args_t {
    std::int32_t slice_factor;  // slice input block into a number of output blocks
    
    // input tensor dims
    std::int32_t num_input_blocks; // can be interpreted as "num blocks", "num tensors", "z-dim", or "batch" loop
    std::int32_t num_tiles_per_input_block;
    std::int32_t input_block_shape_r;
    std::int32_t input_block_shape_c;
    std::int32_t output_block_shape_c;
};

// do this for "num blocks" (per_core_num_blocks) 
//    In0: an RM block with dims (per_core_in_r_tile, input_block_shape_c)
//    Out0: create "slice_factor" output blocks, each with (per_core_in_r_tile, output_block_shape_c)
void hlk_main(tt_core *core_ptr, const hlk_args_t *args) {

    for (int block_index = 0; block_index < args->num_input_blocks; block_index++) { 

        hlk_wait_tiles(core_ptr, HlkOperand::in0, args->num_tiles_per_input_block);

        int in_tile_col_offset = 0;
        for(int slice_idx = 0; slice_idx < args->slice_factor ; ++slice_idx) {
            int in_tile_offset = in_tile_col_offset;
            for(int tile_index_r = 0; tile_index_r < args->input_block_shape_r ; ++tile_index_r) {
                hlk_wait_for_free_tiles(core_ptr, HlkOperand::out0, args->output_block_shape_c);
                for(int tile_index_c = 0; tile_index_c < args->output_block_shape_c ; ++tile_index_c) {
                    int in_tile_index = in_tile_offset + tile_index_c;
                 //   cout << "tile_index_r, tile_index_c, tile_index = " << tile_index_r << " " << tile_index_c << " " << in_tile_index <<  endl;
                    hlk_acquire_dst(core_ptr, DstMode::Half);
                    hlk_copy_tile_to_dst(core_ptr, HlkOperand::in0, in_tile_index, 0);

                    hlk_pack_tile_to_stream(core_ptr, 0, HlkOperand::out0);

                    hlk_release_dst(core_ptr, DstMode::Half);
                }
                in_tile_offset += args->input_block_shape_c;
                hlk_push_tiles(core_ptr, HlkOperand::out0, args->output_block_shape_c);
            }
            in_tile_col_offset += args->output_block_shape_c;
        }

        hlk_pop_tiles(core_ptr, HlkOperand::in0, args->num_tiles_per_input_block);
    }
}
