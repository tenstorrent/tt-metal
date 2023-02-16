#include <cstdint>

#include "compute_hlk_api.h"

struct hlk_args_t {
    std::int32_t per_core_block_cnt;
    std::int32_t per_core_block_size;
};
  
void compute_main(const hlk_args_t *args) {

    for(int block = 0; block < args->per_core_block_cnt; ++block) {

        cb_reserve_back(CB::c_out0, args->per_core_block_size);

        for(int t = 0; t < args->per_core_block_size; ++t)
        {
            acquire_dst(DstMode::Half);

            cb_wait_front(CB::c_in0, 1);
            cb_wait_front(CB::c_in1, 1);

            // ELTWISE_OP is passed in via add_define
            ELTWISE_OP(CB::c_in0, CB::c_in1, 0, 0, 0);
            pack_tile(0, CB::c_out0);

            cb_pop_front(CB::c_in0, 1);
            cb_pop_front(CB::c_in1, 1);

            release_dst(DstMode::Half);
        }

        cb_push_back(CB::c_out0, args->per_core_block_size);
    }
}
