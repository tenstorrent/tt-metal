#include <cstdint>

#include "compute_hlk_api.h"

struct hlk_args_t {
    int num_reductions;

    // per-batch params
    int num_input_blocks;
    int input_block_size;
    int input_block_shape_r;
    int num_input_sub_blocks_c;
    int input_sub_block_shape_c;

    float coefficient;
};

void hlk_main(tt_core *core_ptr, const hlk_args_t *args)
{
    for (int reduction_index = 0; reduction_index < args->num_reductions; reduction_index++) {

        // reduce across blocks in the col dim (they all reduce onto input_block_shape_r)
        for(int in_block_idx=0; in_block_idx < args->num_input_blocks; ++in_block_idx)
        {
            hlk_wait_tiles(core_ptr, HlkOperand::in0, args->input_block_size);

            int input_tile_index = 0;
            for(int r = 0; r < args->input_block_shape_r; ++r)
            {
                bool is_first_block_and_row = in_block_idx == 0 and r == 0;
                for (int c_sub_block_index = 0; c_sub_block_index < args->num_input_sub_blocks_c; ++c_sub_block_index) {

                    hlk_acquire_dst(core_ptr, DstMode::Full);

                    if (not is_first_block_and_row) {
			hlk_copy_tile_to_dst_init(core_ptr);
                        hlk_wait_tiles(core_ptr, HlkOperand::intermed1, args->input_sub_block_shape_c);
                        for (int dst_tile_index = 0; dst_tile_index < args->input_sub_block_shape_c; ++dst_tile_index)
                        {
                            hlk_copy_tile_to_dst(core_ptr, HlkOperand::intermed1, dst_tile_index, dst_tile_index);
                        }
                        hlk_pop_tiles(core_ptr, HlkOperand::intermed1, args->input_sub_block_shape_c);
                    }

                    // reduce a row within a block
                    hlk_reduce_tile_init(core_ptr);
                    for(int dst_tile_index = 0; dst_tile_index < args->input_sub_block_shape_c; ++dst_tile_index)
                    {
                        hlk_reduce_tile(core_ptr, (int)ReduceFunc::Sum, (int)Dim::R, HlkOperand::in0, input_tile_index, dst_tile_index, args->coefficient);
                        input_tile_index++;
                    }

                    // Pack out
                    hlk_wait_for_free_tiles(core_ptr, HlkOperand::intermed1, args->input_sub_block_shape_c);
                    for (int dst_tile_index = 0; dst_tile_index < args->input_sub_block_shape_c; ++dst_tile_index)
                    {
                        hlk_pack_tile_to_stream(core_ptr, dst_tile_index, HlkOperand::intermed1);
                    }
                    hlk_push_tiles(core_ptr, HlkOperand::intermed1, args->input_sub_block_shape_c);

                    hlk_release_dst(core_ptr, DstMode::Full);
                }
            }
            hlk_pop_tiles(core_ptr, HlkOperand::in0, args->input_block_size);
        }
    }

    hlk_add_tile_init(core_ptr);
    for (int reduction_index = 0; reduction_index < args->num_reductions; reduction_index++) {

        for (int c_sub_block_index = 0; c_sub_block_index < args->num_input_sub_blocks_c; ++c_sub_block_index) {
            hlk_acquire_dst(core_ptr, DstMode::Full);

            hlk_wait_tiles(core_ptr, HlkOperand::intermed0, args->input_sub_block_shape_c);
            hlk_wait_tiles(core_ptr, HlkOperand::intermed1, args->input_sub_block_shape_c);

            for (int dst_tile_index = 0; dst_tile_index < args->input_sub_block_shape_c; ++dst_tile_index) {
                hlk_add_tile(
                    core_ptr,
                    HlkOperand::intermed0,
                    HlkOperand::intermed1,
                    dst_tile_index,
                    dst_tile_index,
                    dst_tile_index);
            }

            hlk_pop_tiles(core_ptr, HlkOperand::intermed0, args->input_sub_block_shape_c);
            hlk_pop_tiles(core_ptr, HlkOperand::intermed1, args->input_sub_block_shape_c);

            // Pack out
            hlk_wait_for_free_tiles(core_ptr, HlkOperand::intermed0, args->input_sub_block_shape_c);
            for (int dst_tile_index = 0; dst_tile_index < args->input_sub_block_shape_c; ++dst_tile_index) {
                hlk_pack_tile_to_stream(core_ptr, dst_tile_index, HlkOperand::intermed0);
            }
            hlk_push_tiles(core_ptr, HlkOperand::intermed0, args->input_sub_block_shape_c);

            hlk_release_dst(core_ptr, DstMode::Full);
        }
    }
}
