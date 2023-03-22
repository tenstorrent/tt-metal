#include <cstdint>

#include "llk_3c.h"
//#include "debug_print.h"

namespace NAMESPACE {
void MAIN {
    binary_op_specific_init(ELTWISE_OP_CODE);
    binary_op_init_common(0, 1);

    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_size = get_compile_time_arg_val(1);

    for(uint32_t block = 0; block < per_core_block_cnt; ++block) {

        cb_reserve_back(CB::c_out0, per_core_block_size);

        for(uint32_t t = 0; t < per_core_block_size; ++t)
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

        cb_push_back(CB::c_out0, per_core_block_size);
    }
}
}
