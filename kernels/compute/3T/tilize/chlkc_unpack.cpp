#include <cstdint>
#include "llk_unpack_common.h"
#include "llk_unpack_tilize.h"
namespace NAMESPACE
{

void unpack_main()
{
uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
llk_setup_operands();
llk_unpack_tilize_hw_configure_disaggregated(0);
llk_unpack_tilize_init(0,per_core_block_tile_cnt);
for (uint32_t b = 0U; b < per_core_block_cnt; ++b) {
  llk_wait_tiles(0,per_core_block_tile_cnt);
  for (uint32_t i = 0U; i < per_core_block_tile_cnt; i++) {
    llk_unpack_tilize(0,i,per_core_block_tile_cnt);
  }
  llk_pop_tiles(0,per_core_block_tile_cnt);
}
}
}
