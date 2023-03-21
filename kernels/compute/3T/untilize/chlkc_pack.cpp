#include <cstdint>
#include "llk_pack_common.h"
#include "llk_pack.h"
namespace NAMESPACE
{

void pack_main()
{
uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
llk_pack_init();
llk_pack_hw_configure_disaggregated<false>(16);
llk_setup_outputs();
llk_pack_dest_init<SyncHalf, DstTileFaceLayout::RowMajor, false>();
uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
for (uint32_t b = 0U; b < per_core_block_cnt; ++b) {
  for (uint32_t i = 0U; i < per_core_block_tile_cnt; i++) {
    llk_wait_for_free_tiles<false,false,false>(16,1);
    llk_packer_wait_for_math_done();
    llk_pack<false, SyncHalf, false >(0,16);
    llk_pack_dest_section_done<SyncHalf>();
    llk_push_tiles<false,false>(16,1);
  }
}
}
}
