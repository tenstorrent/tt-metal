#include <cstdint>
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
namespace NAMESPACE
{

struct hlk_args_t
{
int32_t per_core_tile_cnt;
}
;

void math_main()
{
int __outer_loop_iter;
llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, false>();
llk_math_pack_sync_init<SyncHalf>();
constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);
for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
    llk_math_wait_for_dest_available<SyncHalf>();
    llk_math_eltwise_unary_datacopy<A2D, BroadcastType::NONE, SyncHalf>(0);
    llk_math_dest_section_done<SyncHalf>();
}

}
}
