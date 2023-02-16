#include <cstdint>

#include "compute_hlk_api.h"

struct hlk_args_t {
    int block_tile_dim;
    int dst_tile_rows;
    int dst_tile_cols;
    int block_cnt;
    int batch_cnt;
    int in0_block_tile_cnt;
    int in1_block_tile_cnt;
    int out_block_tile_cnt;
    int num_m_sub_blocks;
    int num_n_sub_blocks;
    int num_tiles_per_m_sub_block;
    int num_tiles_per_n_sub_block;
    int num_tiles_per_sub_block;
    int gradient_op;
    int transpose;
};

void hlk_main(tt_core *core_ptr, const hlk_args_t *args)
{

    bool spill = args->block_cnt > 1; //(outer_r>1) or (outer_c>1);
    bool gradient_op = args->gradient_op>0;
    int inner_r = args->num_tiles_per_m_sub_block; // inner row block size in tiles
    int inner_c = args->num_tiles_per_n_sub_block; // inner column block size in tiles
    int inner_d = args->block_tile_dim; // inner block size in tiles
    int outer_r = args->num_m_sub_blocks; // outer row block size (in inner row blocks)
    int outer_c = args->num_n_sub_blocks; // outer column block size (in inner column blocks)
    int outer_id = args->block_cnt; // outer inner dim (in inner dim blocks)
    int in0_block_tile_cnt = inner_r*inner_d*outer_r;
    int in1_block_tile_cnt = inner_c*inner_d*outer_c;
    int out_block_tile_cnt = inner_r * inner_c;

    hlk_mm_tile_init_once(core_ptr, args->transpose);

    for(int batch = 0; batch < args->batch_cnt; batch++)
    {
       bool enable_reload = gradient_op;

       for(int oid = 0; oid < outer_id; oid++)
       {
           bool last_out = oid == (outer_id-1);

           hlk_wait_tiles(core_ptr, HlkOperand::in0, in0_block_tile_cnt);
           hlk_wait_tiles(core_ptr, HlkOperand::in1, in1_block_tile_cnt);

           for (int out_r = 0; out_r < outer_r; out_r++) {

              for (int out_c = 0; out_c < outer_c; out_c++) {

                   hlk_acquire_dst(core_ptr, DstMode::Half);

                   if (enable_reload) {
                       hlk_load_mm_partial_to_dst_init_short(core_ptr, false);
                       hlk_wait_tiles(core_ptr, HlkOperand::intermed0, out_block_tile_cnt);
                       for (int i = 0; i < out_block_tile_cnt; i++) {
                          hlk_load_mm_partial_to_dst(core_ptr, HlkOperand::intermed0, i, i);
                       }
                       hlk_pop_tiles(core_ptr, HlkOperand::intermed0, out_block_tile_cnt);
                       hlk_mm_tile_init_short(core_ptr, args->transpose);
                   }

                   // Compute MM Sub Block
                   int dst_index = 0;
                   for (int in_r = 0; in_r < inner_r; in_r++) {
                       for (int in_c = 0; in_c < inner_c; in_c++) {
                           for (int in_d = 0; in_d < inner_d; in_d++) {
                               int in0_index = out_r*inner_r*inner_d + in_r*inner_d + in_d;
                               int in1_index = out_c*inner_c*inner_d + in_d*inner_c + in_c;
                               hlk_mm_tile(core_ptr, HlkOperand::in0, HlkOperand::in1, in0_index, in1_index, dst_index, args->transpose);
                           }
                           dst_index++;
                       }
                   }

                   if (last_out and !gradient_op) {
                       // Pack out to output buffer
                       hlk_wait_for_free_tiles(core_ptr, HlkOperand::out0, out_block_tile_cnt);
                       for (int i = 0; i < out_block_tile_cnt; i++) {
                          hlk_pack_tile_to_stream(core_ptr, i, HlkOperand::out0);
                       }
                       hlk_push_tiles(core_ptr, HlkOperand::out0, out_block_tile_cnt);
                   } else {
                       // Move partial result to interm buffer
                       hlk_wait_for_free_tiles(core_ptr, HlkOperand::intermed0, out_block_tile_cnt);
                       for (int i = 0; i < out_block_tile_cnt; i++) {
                          hlk_pack_tile_to_stream(core_ptr, i, HlkOperand::intermed0);
                       }
                       hlk_push_tiles(core_ptr, HlkOperand::intermed0, out_block_tile_cnt);
                   }

                   hlk_release_dst(core_ptr, DstMode::Half);
               }
           }
           
           if (spill) enable_reload = true;

           hlk_pop_tiles(core_ptr, HlkOperand::in0, in0_block_tile_cnt);
           hlk_pop_tiles(core_ptr, HlkOperand::in1, in1_block_tile_cnt);

       }
   }

}


