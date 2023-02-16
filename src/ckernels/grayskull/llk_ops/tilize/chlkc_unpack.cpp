#include <cstdint>
#include "llk_unpack_common.h" 
#include "llk_unpack_tilize.h" 
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

void unpack_main(const struct hlk_args_t *args,const int outer_loop_cnt)
{
int __outer_loop_iter;
llk_setup_operands();
llk_unpack_tilize_init();
llk_unpack_tilize_hw_configure_disaggregated(0, args -> per_core_block_c_dim);
for (__outer_loop_iter = 0; __outer_loop_iter < outer_loop_cnt; __outer_loop_iter += 1) {
  for (int i = 0; i < args -> per_core_block_cnt; ++i) {
    llk_wait_blocks(0,1);
    for (int j = 0; j < args -> per_core_block_tile_cnt; ++j) {
      llk_unpack_tilize(0,j,args -> per_core_block_c_dim); 
    }
    llk_pop_blocks(0,1,args -> per_core_block_c_dim);
  }
}
}
}
