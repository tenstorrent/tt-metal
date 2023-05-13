#include <cstdint>

#include "llk_3c.h"

//#include "debug_print.h"

namespace NAMESPACE {
void MAIN {

    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    //UNPACK(( DPRINT << "Block count=" << U32(per_core_block_cnt) << " tile count=" << per_core_block_tile_cnt << ENDL() ));
    tilize_init(CB::c_in0, per_core_block_tile_cnt);

    for(uint32_t b=0;b<per_core_block_cnt;++b)
    {
        cb_wait_front(CB::c_in0, per_core_block_tile_cnt);
        cb_reserve_back(CB::c_out0, per_core_block_tile_cnt);

        tilize_block(CB::c_in0, per_core_block_tile_cnt, CB::c_out0);

        cb_push_back(CB::c_out0, per_core_block_tile_cnt);
        cb_pop_front(CB::c_in0, per_core_block_tile_cnt);
    }
}
}
