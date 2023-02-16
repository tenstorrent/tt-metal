#include <cstdint>
#include "llk_pack_common.h" 
#include "llk_pack.h" 
namespace NAMESPACE
{

struct hlk_args_t 
{
int32_t per_core_tile_cnt; // Total number of tiles produced at the output per core
int32_t per_core_block_cnt; // Number of blocks of size 1xN tiles (1 rows and N cols)
int32_t per_core_block_c_dim; // Block c dim  = (Nx32)
int32_t per_core_block_tile_cnt; // Block tile count = (1xN)
}
;

void pack_main(const struct hlk_args_t *args,const int outer_loop_cnt)
{
int __outer_loop_iter;
llk_pack_init();
llk_pack_hw_configure_disaggregated(16);
llk_setup_outputs();
llk_pack_dest_init<SyncTile16>();
for (__outer_loop_iter = 0; __outer_loop_iter < outer_loop_cnt; __outer_loop_iter += 1) {
  for (int b = 0; b < args -> per_core_tile_cnt; ++b) {
    llk_packer_wait_for_math_done();
    llk_wait_for_free_tiles(16,1);
    llk_pack<false, SyncTile16>(0,16);
    llk_push_tiles(16,1);
    llk_pack_dest_section_done<SyncTile16>();
  }
}
}
}
