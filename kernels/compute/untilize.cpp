#include <cstdint>

#include "compute_hlk_api.h"

void compute_main() {

    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);

    for(uint32_t b=0;b<per_core_block_cnt;++b)
    {
        cb_wait_front(CB::c_in0, per_core_block_tile_cnt);

        for (uint32_t i = 0; i < per_core_block_tile_cnt; i++) {
            cb_reserve_back(CB::c_out0, 1);
            acquire_dst(DstMode::Half);
            untilize_and_copy(CB::c_in0, i, 0, per_core_block_tile_cnt);
            pack_tile(0, CB::c_out0);
            release_dst(DstMode::Half);
            cb_push_back(CB::c_out0, 1);
        }

        cb_pop_front(CB::c_in0, per_core_block_tile_cnt);

    }
}
