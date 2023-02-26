#include <cstdint>

#include "compute_hlk_api.h"

struct hlk_args_t {
    std::int32_t block_tile_dim;
    std::int32_t block_cnt;
    std::int32_t batch_cnt;
    std::int32_t num_m_sub_blocks;
    std::int32_t num_n_sub_blocks;
    std::int32_t num_tiles_per_m_sub_block;
    std::int32_t num_tiles_per_n_sub_block;
    std::int32_t gradient_op;
    std::int32_t transpose;
};

void hlk_main(tt_core *core_ptr, const hlk_args_t *args) {
    for(int block = 0; block < args->block_cnt; ++block) {
       hlk_acquire_dst(core_ptr, DstMode::Half);

       // Wait for tiles on the input
       hlk_wait_tiles(core_ptr, HlkOperand::in0, args->block_tile_dim);

       for(int t = 0; t < args->block_tile_dim; ++t) {
           hlk_copy_tile_to_dst(core_ptr, HlkOperand::in0, t, t);
           hlk_sfpu_sqrt(core_ptr, t);
       }
       // Pop input and push to output 
       hlk_pop_tiles(core_ptr, HlkOperand::in0, args->block_tile_dim);

       // Wait for space in output
       hlk_wait_for_free_tiles(core_ptr, HlkOperand::out0, args->block_tile_dim);

       for(int t = 0; t < args->block_tile_dim; ++t) {
           hlk_pack_tile_to_stream(core_ptr, t, HlkOperand::out0);
       }

       hlk_push_tiles(core_ptr, HlkOperand::out0, args->block_tile_dim);
       hlk_release_dst(core_ptr, DstMode::Half);
    }
}
