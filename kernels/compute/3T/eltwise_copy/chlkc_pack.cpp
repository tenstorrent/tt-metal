#include <cstdint>
#include "llk_pack_common.h"
#include "llk_pack.h"
namespace NAMESPACE
{

void pack_main()
{
int __outer_loop_iter;
llk_pack_init();
llk_pack_hw_configure_disaggregated<false>(16);
llk_setup_outputs();
llk_pack_dest_init<SyncHalf, DstTileFaceLayout::RowMajor, false>();
constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);
for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
    llk_packer_wait_for_math_done();
    llk_wait_for_free_tiles<false,false,false>(16,1);
    llk_pack<false, SyncHalf, false >(0,16);
    llk_push_tiles<false,false>(16,1);
    llk_pack_dest_section_done<SyncHalf>();
}
}
}
