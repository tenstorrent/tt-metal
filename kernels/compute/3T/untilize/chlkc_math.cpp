#include <cstdint>
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
namespace NAMESPACE
{

void math_main()
{
uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, false>();
llk_math_pack_sync_init<SyncHalf>();
uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
for (uint32_t b = 0U; b < per_core_block_cnt; ++b) {
  for (uint32_t i = 0U; i < per_core_block_tile_cnt; i++) {
    llk_math_wait_for_dest_available<SyncHalf>();
    llk_math_eltwise_unary_datacopy<A2D, BroadcastType::NONE, SyncHalf>(0);
    llk_math_dest_section_done<SyncHalf>();
  }
}
}
}
