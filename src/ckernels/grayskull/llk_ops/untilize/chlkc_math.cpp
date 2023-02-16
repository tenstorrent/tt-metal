#include <cstdint>
#include "llk_math_common.h" 
#include "llk_math_eltwise_unary_datacopy.h" 
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

void math_main(const struct hlk_args_t *args,const int outer_loop_cnt)
{
int __outer_loop_iter;
llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, false>();
llk_math_pack_sync_init<SyncTile16>();
for (__outer_loop_iter = 0; __outer_loop_iter < outer_loop_cnt; __outer_loop_iter += 1) {
  for (int b = 0; b < args -> per_core_tile_cnt; ++b) {
    llk_math_wait_for_dest_available<SyncTile16>();
    llk_math_eltwise_unary_datacopy<A2D, BroadcastType::NONE, SyncTile16>(0);
    llk_math_dest_section_done<SyncTile16>();
  }
}
}
}
