#include <cstdint>

#include "compute_hlk_api.h"

void compute_main() {
    uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);
    for(uint32_t b=0;b<per_core_tile_cnt;++b)
    {
        acquire_dst(DstMode::Half);

        // Pop tile after tile, copy to DST and pack
        cb_wait_front(CB::c_in0, 1);
        cb_reserve_back(CB::c_out0, 1);

        copy_tile(CB::c_in0, 0, 0);
        pack_tile(0, CB::c_out0);

        cb_pop_front(CB::c_in0, 1);
        cb_push_back(CB::c_out0, 1);

        release_dst(DstMode::Half);
    }
}
