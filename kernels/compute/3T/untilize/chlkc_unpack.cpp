#include <cstdint>
#include "llk_unpack_common.h"
#include "llk_unpack_untilize.h"
namespace NAMESPACE
{

void unpack_main()
{
uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
llk_setup_operands();
llk_unpack_untilize_hw_configure_disaggregated(0);
llk_unpack_untilize_init(0);
for (uint32_t b = 0U; b < per_core_block_cnt; ++b) {
  llk_wait_tiles(0,per_core_block_tile_cnt);
  for (uint32_t i = 0U; i < per_core_block_tile_cnt; i++) {
    llk_unpack_untilize(0,i,per_core_block_tile_cnt);
  }
  llk_pop_tiles(0,per_core_block_tile_cnt);
}
}
}
