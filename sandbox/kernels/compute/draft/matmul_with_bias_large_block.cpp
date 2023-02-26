#include <cstdint>

#include "compute_hlk_api.h"

constexpr bool hack = true;

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
};

void hlk_main(tt_core *core_ptr, const hlk_args_t *args)
{

    hlk_wait_for_free_tiles(core_ptr, HlkOperand::intermed1, args->out_block_tile_cnt);

    for(int block_index = 0; block_index < args->block_cnt; ++block_index)
    {
        bool is_first_block = block_index == 0;
        bool is_last_block = block_index == args->block_cnt - 1;

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
if (not hack) {
                    hlk_wait_tiles(core_ptr, HlkOperand::intermed0, args->num_tiles_per_sub_block);
}
                    hlk_load_mm_partial_to_dst_init(core_ptr);
                    // Move Sub Block from intermediate buffer to DST
                    for (int m = 0; m < args->num_tiles_per_m_sub_block; ++m) {
                        for (int n = 0; n < args->num_tiles_per_n_sub_block; ++n) {
if (hack) {
                            int out_tile_offset = (m_sub_block_offset + m) * args->dst_tile_cols + n_sub_block_offset + n;
                            hlk_load_mm_partial_to_dst(core_ptr, HlkOperand::intermed0, out_tile_offset, dst_tile_index);
} else {
                            hlk_load_mm_partial_to_dst(core_ptr, HlkOperand::intermed0, dst_tile_index, dst_tile_index);
}
                            dst_tile_index++;
                        }
                    }
if (not hack) {
                    hlk_pop_tiles(core_ptr, HlkOperand::intermed0, args->num_tiles_per_sub_block);
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
                            hlk_mm_tile(core_ptr, HlkOperand::in0, HlkOperand::in1, in0_index, in1_index, dst_tile_index,false);
                        }
                        dst_tile_index++;
                    }
                }

                if (is_last_block) {
                    // Pack out to output buffer
                    dst_tile_index = 0;
                    for (int m = 0; m < args->num_tiles_per_m_sub_block; ++m) {
                        for (int n = 0; n < args->num_tiles_per_n_sub_block; ++n) {
                            int out_tile_offset = (m_sub_block_offset + m) * args->dst_tile_cols + n_sub_block_offset + n;
                            hlk_pack_tile_to_stream(core_ptr, dst_tile_index, HlkOperand::intermed1, out_tile_offset);
                            dst_tile_index++;
                        }
                    }
                } else {
                    // Move Sub Block from DST to intermediate buffer
                    dst_tile_index = 0;
                    hlk_wait_for_free_tiles(core_ptr, HlkOperand::intermed0, args->num_tiles_per_sub_block);
                    for (int m = 0; m < args->num_tiles_per_m_sub_block; ++m) {
                        for (int n = 0; n < args->num_tiles_per_n_sub_block; ++n) {
if (hack) {
                            int out_tile_offset = (m_sub_block_offset + m) * args->dst_tile_cols + n_sub_block_offset + n;
                            hlk_pack_tile_to_stream(core_ptr, dst_tile_index, HlkOperand::intermed0, out_tile_offset);
} else {
                            hlk_pack_tile_to_stream(core_ptr, dst_tile_index, HlkOperand::intermed0, dst_tile_index);
}
                            dst_tile_index++;
                        }
                    }
                    hlk_push_tiles(core_ptr, HlkOperand::intermed0, args->num_tiles_per_sub_block);
                }

                hlk_release_dst(core_ptr, DstMode::Full);
                n_sub_block_offset += args->num_tiles_per_n_sub_block;
            }
            m_sub_block_offset += args->num_tiles_per_m_sub_block;
        }

        hlk_pop_tiles(core_ptr, HlkOperand::in0, args->in0_block_tile_cnt);
        hlk_pop_tiles(core_ptr, HlkOperand::in1, args->in1_block_tile_cnt);

    }

    hlk_push_tiles(core_ptr, HlkOperand::intermed1, args->out_block_tile_cnt);


    hlk_add_tile_bcast_init(core_ptr);
    hlk_wait_tiles(core_ptr, HlkOperand::in2, args->dst_tile_cols);
    hlk_wait_tiles(core_ptr, HlkOperand::intermed1, args->out_block_tile_cnt);
    int m_sub_block_tile_offset = 0;
    hlk_wait_for_free_tiles(core_ptr, HlkOperand::out0, args->out_block_tile_cnt);
    for(int m_sub_block_index = 0; m_sub_block_index < args->num_m_sub_blocks; ++m_sub_block_index) {
        int n_sub_block_offset = 0;
        for (int n_sub_block_index = 0; n_sub_block_index < args->num_n_sub_blocks; ++n_sub_block_index) {
            hlk_acquire_dst(core_ptr, DstMode::Full);
            int dst_tile_index = 0;
            for (int m = 0; m < args->num_tiles_per_m_sub_block; ++m) {
                int n_tile_offset = m * args->dst_tile_cols;
                for (int n = 0; n < args->num_tiles_per_n_sub_block; ++n) {
                    int out_tile_offset = m_sub_block_tile_offset + n_tile_offset + n_sub_block_offset + n;
                    hlk_add_tile_bcast(core_ptr, (int)Dim::R, HlkOperand::intermed1, HlkOperand::in2, out_tile_offset, n + n_sub_block_offset, dst_tile_index);
                    dst_tile_index++;
                }
            }
            dst_tile_index = 0;
            for (int m = 0; m < args->num_tiles_per_m_sub_block; ++m) {
                int n_tile_offset = m * args->dst_tile_cols;
                for (int n = 0; n < args->num_tiles_per_n_sub_block; ++n) {
                    int out_tile_offset = m_sub_block_tile_offset + n_tile_offset + n_sub_block_offset + n;
                    hlk_pack_tile_to_stream(core_ptr, dst_tile_index, HlkOperand::out0, out_tile_offset);
                    dst_tile_index++;
                }
            }
            hlk_release_dst(core_ptr, DstMode::Full);
            n_sub_block_offset += args->num_tiles_per_n_sub_block;
        }
        m_sub_block_tile_offset += args->num_tiles_per_m_sub_block * args->dst_tile_cols;
    }
    hlk_pop_tiles(core_ptr, HlkOperand::in2, args->dst_tile_cols);
    hlk_pop_tiles(core_ptr, HlkOperand::intermed1, args->out_block_tile_cnt);
    hlk_push_tiles(core_ptr, HlkOperand::out0, args->out_block_tile_cnt);
}


