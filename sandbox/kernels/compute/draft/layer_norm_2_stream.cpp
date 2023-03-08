#include <cstdint>
#include "compute_hlk_api.h"

struct hlk_args_t {
    std::int32_t batch_cnt;
    std::int32_t num_m_sub_blocks;
    std::int32_t num_n_sub_blocks;
};

void hlk_main(tt_core *core_ptr, const hlk_args_t *args) {

  hlk_multiply_tile_init_once(core_ptr);
  for(int batch = 0; batch < args->batch_cnt; ++batch) {


    bool enable_reload[8] = {false, false, false, false, false, false, false, false};

    for(int m_block = 0; m_block < args->num_m_sub_blocks; ++m_block) {
    for(int n_block = 0; n_block < args->num_n_sub_blocks; ++n_block) {


      // -----------------------------
      // OP: layernorm_39.dc.multiply.8
      // -----------------------------

      hlk_multiply_tile_init_short(core_ptr);
      hlk_wait_tiles(core_ptr, HlkOperand::in0, 8);
      hlk_wait_tiles(core_ptr, HlkOperand::in1, 8);

      hlk_acquire_dst(core_ptr, DstMode::Half);

      for(int t = 0; t < 8; ++t) {
         hlk_multiply_tile(core_ptr, HlkOperand::in0, HlkOperand::in1, t, t, t);
      }

      hlk_pop_tiles(core_ptr, HlkOperand::in0, 8);
      hlk_pop_tiles(core_ptr, HlkOperand::in1, 8);

      // -----------------------------
      // OP: layernorm_39.dc.multiply.9
      // -----------------------------

      hlk_multiply_tile_from_dst_init_short(core_ptr);
      hlk_wait_tiles(core_ptr, HlkOperand::in2, 8);

      for(int t = 0; t < 8; ++t) {
         hlk_multiply_tile_from_dst(core_ptr, HlkOperand::in2, t, t);
      }

      hlk_pop_tiles(core_ptr, HlkOperand::in2, 8);

      // -----------------------------
      // OP: layernorm_39.dc.add.10
      // -----------------------------

      hlk_add_tile_from_dst_init_short(core_ptr);
      hlk_wait_tiles(core_ptr, HlkOperand::in3, 8);

      for(int t = 0; t < 8; ++t) {
         hlk_add_tile_from_dst(core_ptr, HlkOperand::in3, t, t);
      }

      hlk_pop_tiles(core_ptr, HlkOperand::in3, 8);

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
