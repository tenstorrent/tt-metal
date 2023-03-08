#include <cstdint>

#include "compute_hlk_api.h"

struct hlk_args_t {
    int batch_size;
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
};

void hlk_main(tt_core *core_ptr, const hlk_args_t *args)
{
    int batch_offset = 0;
    for (int batch_index = 0; batch_index < args->batch_size; batch_index++)
    {
        for(int block_index = 0; block_index < args->block_cnt; ++block_index)
        {
            hlk_mm_tile_init(core_ptr, false);

            hlk_wait_tiles(core_ptr, CB::c_in0, args->in0_block_tile_cnt);
            hlk_wait_tiles(core_ptr, CB::c_in1, args->in1_block_tile_cnt);

            //hlk_wait_for_free_tiles(core_ptr, CB::c_out0, args->out_block_tile_cnt);

            int m_sub_block_offset = 0;
            for(int m_sub_block_index = 0; m_sub_block_index < args->num_m_sub_blocks; ++m_sub_block_index) {
                int n_sub_block_offset = 0;
                for (int n_sub_block_index = 0; n_sub_block_index < args->num_n_sub_blocks; ++n_sub_block_index) {

                    hlk_acquire_dst(core_ptr, DstMode::Full);

                    int dst_tile_index = 0;
                    if (block_index > 0) {
                        hlk_copy_tile_to_dst_init(core_ptr);
                        // Move Sub Block from intermediate buffer to DST
                        int m_tile_offset = 0;
                        for (int m = 0; m < args->num_tiles_per_m_sub_block; ++m) {
                            for (int n = 0; n < args->num_tiles_per_n_sub_block; ++n) {
                                int out_tile_offset = batch_offset + (m_sub_block_offset + m) * args->dst_tile_cols + n_sub_block_offset + n;
                                //int out_tile_offset = m_sub_block_offset + n_sub_block_offset + m_tile_offset + n;
                                hlk_copy_tile_to_dst(core_ptr, CB::c_intermed0, out_tile_offset, dst_tile_index);
                                dst_tile_index++;
                            }
                            m_tile_offset+=args->dst_tile_cols;
                        }
                        hlk_mm_tile_init(core_ptr, false);
                    }

                    // Compute MM Sub Block
                    dst_tile_index = 0;
                    for (int m = 0; m < args->num_tiles_per_m_sub_block; ++m) {
                        for (int n = 0; n < args->num_tiles_per_n_sub_block; ++n) {
                            for (int k = 0; k < args->block_tile_dim; ++k) {
                                int in0_index = (m_sub_block_offset + m) * args->block_tile_dim + k;
                                int in1_index = k * args->dst_tile_cols + n_sub_block_offset + n;
                                hlk_mm_tile(core_ptr, CB::c_in0, CB::c_in1, in0_index, in1_index, dst_tile_index,false);
                            }
                            dst_tile_index++;
                        }
                    }

                    if (block_index == args->block_cnt - 1) {
                        // Pack out to output buffer
                        dst_tile_index = 0;
                        int m_tile_offset = 0;
                        hlk_wait_for_free_tiles(core_ptr, CB::c_out0, args->num_tiles_per_sub_block);
                        for (int m = 0; m < args->num_tiles_per_m_sub_block; ++m) {
                            for (int n = 0; n < args->num_tiles_per_n_sub_block; ++n) {
                                //int out_tile_offset = m_sub_block_offset + n_sub_block_offset + m_tile_offset + n;
                                int out_tile_offset = batch_offset + (m_sub_block_offset + m) * args->dst_tile_cols + n_sub_block_offset + n;
                                hlk_pack_tile_to_stream(core_ptr, dst_tile_index, CB::c_out0, out_tile_offset);
                                dst_tile_index++;
                            }
                            m_tile_offset+=args->dst_tile_cols;
                        }
                        hlk_push_tiles(core_ptr, CB::c_out0, args->num_tiles_per_sub_block);
                    } else {
                        // Move Sub Block from DST to intermediate buffer
                        hlk_wait_for_free_tiles(core_ptr, CB::c_intermed0, args->num_tiles_per_sub_block);
                        dst_tile_index = 0;
                        int m_tile_offset = 0;
                        for (int m = 0; m < args->num_tiles_per_m_sub_block; ++m) {
                            for (int n = 0; n < args->num_tiles_per_n_sub_block; ++n) {
                                //int out_tile_offset = m_sub_block_offset + n_sub_block_offset + m_tile_offset + n;
                                int out_tile_offset = batch_offset + (m_sub_block_offset + m) * args->dst_tile_cols + n_sub_block_offset + n;
                                hlk_pack_tile_to_stream(core_ptr, dst_tile_index, CB::c_intermed0, out_tile_offset);
                                dst_tile_index++;
                            }
                            m_tile_offset+=args->dst_tile_cols;
                        }
                        hlk_push_tiles(core_ptr, CB::c_intermed0, args->num_tiles_per_sub_block);
                    }

                    hlk_release_dst(core_ptr, DstMode::Full);
                    n_sub_block_offset += args->num_tiles_per_n_sub_block;
                }
                m_sub_block_offset += args->num_tiles_per_m_sub_block;
                //m_sub_block_offset += (args->num_tiles_per_sub_block*args->num_n_sub_blocks);
            }

            hlk_pop_tiles(core_ptr, CB::c_in0, args->in0_block_tile_cnt);
            hlk_pop_tiles(core_ptr, CB::c_in1, args->in1_block_tile_cnt);

            //hlk_push_tiles(core_ptr, CB::c_out0, args->out_block_tile_cnt);
        }
        batch_offset += args->num_m_sub_blocks * args->num_tiles_per_m_sub_block * args->num_n_sub_blocks * args->num_tiles_per_n_sub_block;
    }
}
