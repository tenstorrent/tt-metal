#include <cstdint>
#include "llk_unpack_common.h" 
#include "llk_unpack_untilize.h" 
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

void unpack_main(const struct hlk_args_t *args,const int outer_loop_cnt)
{
int __outer_loop_iter;
llk_setup_operands();
llk_unpack_untilize_init();
llk_unpack_untilize_hw_configure_disaggregated(0);
for (__outer_loop_iter = 0; __outer_loop_iter < outer_loop_cnt; __outer_loop_iter += 1) {
  for (int i = 0; i < args -> per_core_block_cnt; ++i) {
    for (int j = 0; j < args -> per_core_block_tile_r_dim; ++j) {
      llk_wait_tiles(0,args -> per_core_block_tile_c_dim);
      llk_unpack_untilize<true>(0,args -> per_core_block_tile_c_dim); 
      llk_unpack_untilize<false>(0,args -> per_core_block_tile_c_dim); 
      llk_pop_tiles(0,args -> per_core_block_tile_c_dim);
    }
  }
}
}
}
