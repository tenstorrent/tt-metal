#include <cstdint>

#include "compute_hlk_api.h"

struct hlk_args_t {
    std::int32_t per_core_tile_cnt;
};
  
void compute_main(const hlk_args_t *args) {
    for(int b=0;b<args->per_core_tile_cnt;++b)
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
