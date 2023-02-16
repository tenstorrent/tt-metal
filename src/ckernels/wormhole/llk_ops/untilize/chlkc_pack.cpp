#include <cstdint>
#include "llk_pack_common.h" 
#include "llk_pack.h" 
namespace NAMESPACE
{

struct hlk_args_t 
{
int32_t per_core_tile_cnt; // Total number of tiles produced at the output per core
int32_t per_core_block_tile_r_dim; // Block tile r dim (RT)
int32_t per_core_block_tile_c_dim; // Block tile c dim (CT)
int32_t per_core_block_cnt; // Number of blocks of size (RTxCT)
}
;

void pack_main(const struct hlk_args_t *args,const int outer_loop_cnt)
{
int __outer_loop_iter;
llk_pack_init();
llk_pack_hw_configure_disaggregated<true>(16);
llk_setup_outputs();
llk_pack_dest_init<SyncTile16>();
for (__outer_loop_iter = 0; __outer_loop_iter < outer_loop_cnt; __outer_loop_iter += 1) {
  for (int b = 0; b < args -> per_core_tile_cnt; ++b) {
    llk_packer_wait_for_math_done();
    llk_wait_for_free_blocks(16,1);
    llk_pack<false, SyncTile16, true>(0,16);
    llk_push_blocks(16,1);
    llk_pack_dest_section_done<SyncTile16>();
  }
}
}
}
