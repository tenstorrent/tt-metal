#include <cstdint>

#include "compute_hlk_api.h"

struct hlk_args_t
{
int32_t per_core_block_cnt; // Number of blocks of size 1xN tiles (1 rows and N cols)
int32_t per_core_block_tile_cnt; // Block tile count = (1xN)
};

void compute_main(const hlk_args_t *args) {

    for(int b=0;b<args->per_core_block_cnt;++b)
    {
        cb_wait_front(CB::c_in0, args->per_core_block_tile_cnt);

        for (int i = 0; i < args->per_core_block_tile_cnt; i++) {
            cb_reserve_back(CB::c_out0, 1);
            acquire_dst(DstMode::Half);
            untilize_and_copy(CB::c_in0, i, 0, args->per_core_block_tile_cnt);
            pack_tile(0, CB::c_out0);
            release_dst(DstMode::Half);
            cb_push_back(CB::c_out0, 1);
        }

        cb_pop_front(CB::c_in0, args->per_core_block_tile_cnt);

    }
}
