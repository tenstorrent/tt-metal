#include <cstdint>

#include "compute_hlk_api.h"

struct hlk_args_t {
    int block_tile_dim;
    int dst_tile_rows;
    int dst_tile_cols;
    int block_cnt;
    int in0_block_tile_cnt;
    int in1_block_tile_cnt;
    int out_block_tile_cnt;
    int num_m_sub_blocks;
    int num_n_sub_blocks;
    int num_tiles_per_m_sub_block;
    int num_tiles_per_n_sub_block;
    int num_tiles_per_sub_block;
    int gradient_op;
};

void hlk_main(tt_core *core_ptr, const hlk_args_t *args)
{

    for(int block_index = 0; block_index < args->block_cnt; ++block_index)
    {
        bool is_first_block = block_index == 0;

        hlk_mm_tile_init(core_ptr, false);

        hlk_wait_tiles(core_ptr, HlkOperand::in0, args->in0_block_tile_cnt);
        hlk_wait_tiles(core_ptr, HlkOperand::in1, args->in1_block_tile_cnt);

        int m_sub_block_offset = 0;
        for(int m_sub_block_index = 0; m_sub_block_index < args->num_m_sub_blocks; ++m_sub_block_index) {
            int n_sub_block_offset = 0;
            for (int n_sub_block_index = 0; n_sub_block_index < args->num_n_sub_blocks; ++n_sub_block_index) {

                hlk_acquire_dst(core_ptr, DstMode::Full);

                int dst_tile_index = 0;
                if (not is_first_block) {
                    hlk_wait_tiles(core_ptr, HlkOperand::intermed1, args->num_tiles_per_sub_block);
                    hlk_load_mm_partial_to_dst_init(core_ptr);
                    // Move Sub Block from intermediate buffer to DST
                    for (int m = 0; m < args->num_tiles_per_m_sub_block; ++m) {
                        for (int n = 0; n < args->num_tiles_per_n_sub_block; ++n) {
                            hlk_load_mm_partial_to_dst(core_ptr, HlkOperand::intermed1, dst_tile_index, dst_tile_index);
                            dst_tile_index++;
                        }
                    }
                    hlk_pop_tiles(core_ptr, HlkOperand::intermed1, args->num_tiles_per_sub_block);
                    hlk_mm_tile_init(core_ptr, false);
                }

                // Compute MM Sub Block
                dst_tile_index = 0;
                for (int m = 0; m < args->num_tiles_per_m_sub_block; ++m) {
                    for (int n = 0; n < args->num_tiles_per_n_sub_block; ++n) {
                        for (int k = 0; k < args->block_tile_dim; ++k) {
                            int in0_index = (m_sub_block_offset + m) * args->block_tile_dim + k;
                            int in1_index = k * args->dst_tile_cols + n_sub_block_offset + n;
                            hlk_mm_tile(core_ptr, HlkOperand::in0, HlkOperand::in1, in0_index, in1_index, dst_tile_index,false);
                        }
                        dst_tile_index++;
                    }
                }

                // Move Sub Block from DST to intermediate buffer
                dst_tile_index = 0;
                hlk_wait_for_free_tiles(core_ptr, HlkOperand::intermed1, args->num_tiles_per_sub_block);
                for (int m = 0; m < args->num_tiles_per_m_sub_block; ++m) {
                    for (int n = 0; n < args->num_tiles_per_n_sub_block; ++n) {
                        hlk_pack_tile_to_stream(core_ptr, dst_tile_index, HlkOperand::intermed1);
                        dst_tile_index++;
                    }
                }

                hlk_release_dst(core_ptr, DstMode::Full);

                hlk_push_tiles(core_ptr, HlkOperand::intermed1, args->num_tiles_per_sub_block);

                n_sub_block_offset += args->num_tiles_per_n_sub_block;
            }
            m_sub_block_offset += args->num_tiles_per_m_sub_block;
        }

        hlk_pop_tiles(core_ptr, HlkOperand::in0, args->in0_block_tile_cnt);
        hlk_pop_tiles(core_ptr, HlkOperand::in1, args->in1_block_tile_cnt);
       
    }

    /* Finish Computing Matmul */

    hlk_wait_tiles(core_ptr, HlkOperand::intermed1, args->out_block_tile_cnt);

    /* Add Matmul result to the Accumulator */
    hlk_add_tile_init(core_ptr);
    int dst_r_offset = 0;
    int dst_c_offset = 0;
    for(int dst_r = 0; dst_r < args->dst_tile_rows/args->num_tiles_per_m_sub_block; ++dst_r) {
	    dst_c_offset = 0;
        for (int dst_c = 0; dst_c < args->dst_tile_cols/args->num_tiles_per_n_sub_block; ++dst_c) {
            hlk_acquire_dst(core_ptr, DstMode::Full);

            int dst_tile_index = 0;
            hlk_wait_tiles(core_ptr, HlkOperand::intermed0, args->num_tiles_per_sub_block);

            for (int m = 0; m < args->num_tiles_per_m_sub_block; ++m) {
               for (int n = 0; n < args->num_tiles_per_n_sub_block; ++n) {
		            int intermed1_buffer_tile_offset = dst_r_offset +
		                ((dst_c_offset + m)%args->num_n_sub_blocks)*args->num_tiles_per_sub_block +
			            ((dst_c_offset + m)/args->num_n_sub_blocks)*args->num_tiles_per_n_sub_block + n;
	                                                
                   hlk_add_tile(core_ptr, HlkOperand::intermed0, HlkOperand::intermed1, dst_tile_index, intermed1_buffer_tile_offset, dst_tile_index);
                   dst_tile_index++;
	       }    
	    }   

            hlk_pop_tiles(core_ptr, HlkOperand::intermed0, args->num_tiles_per_sub_block);

            dst_tile_index = 0;
            hlk_wait_for_free_tiles(core_ptr, HlkOperand::intermed0, args->num_tiles_per_sub_block);
            for (int m = 0; m < args->num_tiles_per_m_sub_block; ++m) {
                for (int n = 0; n < args->num_tiles_per_n_sub_block; ++n) {
                    hlk_pack_tile_to_stream(core_ptr, dst_tile_index, HlkOperand::intermed0);
                    dst_tile_index++;
                }
            }

            hlk_release_dst(core_ptr, DstMode::Full);

            hlk_push_tiles(core_ptr, HlkOperand::intermed0, args->num_tiles_per_sub_block);

            dst_c_offset += args->num_tiles_per_m_sub_block;
        }
        dst_r_offset += (args->num_n_sub_blocks*args->num_tiles_per_sub_block);
    }

    hlk_pop_tiles(core_ptr, HlkOperand::intermed1, args->out_block_tile_cnt);

}


