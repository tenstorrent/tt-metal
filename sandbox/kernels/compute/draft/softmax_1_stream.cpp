#include <cstdint>
#include "compute_hlk_api.h"

struct hlk_args_t {
    std::int32_t batch_cnt;
    std::int32_t num_m_sub_blocks;
    std::int32_t num_n_sub_blocks;
};

void hlk_main(tt_core *core_ptr, const hlk_args_t *args) {

  hlk_mm_tile_init_once(core_ptr, false);
  for(int batch = 0; batch < args->batch_cnt; ++batch) {


    bool enable_reload[8] = {false, false, false, false, false, false, false, false};

    for(int m_block = 0; m_block < args->num_m_sub_blocks; ++m_block) {
    for(int n_block = 0; n_block < args->num_n_sub_blocks; ++n_block) {

      if ((m_block < 2) &&
          (n_block < 1)) {

      // -----------------------------
      // OP: softmax_19.dc.reduce_sum.1.lc1
      // -----------------------------

      hlk_mm_tile_init(core_ptr, false);
      hlk_wait_tiles(core_ptr, HlkOperand::in0, 8);
      if (m_block == 0) {
        hlk_wait_tiles(core_ptr, HlkOperand::in1, 4);
      }

      hlk_acquire_dst(core_ptr, DstMode::Half);

      int dst_index_0 = 0;
      for (int in_r = 0; in_r < 2; in_r++) {
          for (int in_c = 0; in_c < 1; in_c++) {
              for (int in_d = 0; in_d < 4; in_d++) {
                  int in0_index = in_r*4 + in_d;
                  int in1_index = in_d*1 + in_c;
                  hlk_mm_tile(core_ptr, HlkOperand::in0 , HlkOperand::in1, in0_index, in1_index, dst_index_0, false);
              }
              dst_index_0++;
          }
      }

      hlk_pop_tiles(core_ptr, HlkOperand::in0, 8);
      if ((m_block == 1)) {
        hlk_pop_tiles(core_ptr, HlkOperand::in1, 4);
      }

      hlk_wait_for_free_tiles(core_ptr, HlkOperand::intermed0, 2);
      for(int t = 0; t < 2 ; ++t) {
         hlk_pack_tile_to_stream(core_ptr, t, HlkOperand::intermed0);
      }
      hlk_push_tiles(core_ptr, HlkOperand::intermed0, 2);

      hlk_release_dst(core_ptr, DstMode::Half);

      } // end for matmul reduce mblock dim check


      // -----------------------------
      // OP: softmax_19.dc.reciprocal.2
      // -----------------------------

      hlk_copy_tile_to_dst_init(core_ptr);
      hlk_sfpu_reciprocal_init(core_ptr);
      hlk_wait_tiles(core_ptr, HlkOperand::intermed0, 2);

      hlk_acquire_dst(core_ptr, DstMode::Half);

      for(int t = 0; t < 2; ++t) {
        hlk_copy_tile_to_dst(core_ptr, HlkOperand::intermed0, t, t);
        hlk_sfpu_reciprocal(core_ptr, t);
      }

      hlk_pop_tiles(core_ptr, HlkOperand::intermed0, 2);

      hlk_wait_for_free_tiles(core_ptr, HlkOperand::intermed1, 2);
      for(int t = 0; t < 2 ; ++t) {
         hlk_pack_tile_to_stream(core_ptr, t, HlkOperand::intermed1);
      }
      hlk_push_tiles(core_ptr, HlkOperand::intermed1, 2);

      hlk_release_dst(core_ptr, DstMode::Half);


      // -----------------------------
      // OP: softmax_19.dc.multiply.3
      // -----------------------------

      hlk_multiply_tile_init_short(core_ptr);
      hlk_wait_tiles(core_ptr, HlkOperand::in2, 8);
      hlk_wait_tiles(core_ptr, HlkOperand::intermed1, 2);

      hlk_acquire_dst(core_ptr, DstMode::Half);

      for(int t = 0; t < 8; ++t) {
         hlk_multiply_tile(core_ptr, HlkOperand::in2, HlkOperand::intermed1, t, t/4, t);
      }

      hlk_pop_tiles(core_ptr, HlkOperand::in2, 8);
      hlk_pop_tiles(core_ptr, HlkOperand::intermed1, 2);

      hlk_wait_for_free_tiles(core_ptr, HlkOperand::out0, 8);
      for(int t = 0; t < 8 ; ++t) {
         hlk_pack_tile_to_stream(core_ptr, t, HlkOperand::out0);
      }
      hlk_push_tiles(core_ptr, HlkOperand::out0, 8);

      hlk_release_dst(core_ptr, DstMode::Half);

    } // n_block loop end

    } // m_block loop end

  } // batch loop end

}
