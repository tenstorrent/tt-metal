#include <cstdint>
#include "llk_unpack_common.h"
#include "llk_unpack_A.h"
namespace NAMESPACE
{

void unpack_main()
{
int __outer_loop_iter;
llk_setup_operands();
llk_unpack_A_init<BroadcastType::NONE, false, false>();
llk_unpack_A_hw_configure_disaggregated<BroadcastType::NONE, false, false, false>(0);
constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);
for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
    llk_wait_tiles(0,1);
    llk_unpack_A(0,0);
    llk_pop_tiles(0,1);
}
}
}
