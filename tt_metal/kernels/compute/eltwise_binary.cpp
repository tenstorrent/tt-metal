// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"

#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

ALWI void ACQ() { acquire_dst(tt::DstMode::Half); }
ALWI void REL() { release_dst(tt::DstMode::Half); }

namespace NAMESPACE {
void MAIN {

    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_size = get_compile_time_arg_val(1);

    #ifdef SFPU_OP_INIT_PRE_IN0_0
        constexpr auto cb_inp0 = tt::CB::c_intermed0;
    #else
        constexpr auto cb_inp0 = tt::CB::c_in0;
    #endif

    #ifdef SFPU_OP_INIT_PRE_IN1_0
        constexpr auto cb_inp1 = tt::CB::c_intermed1;
    #else
        constexpr auto cb_inp1 = tt::CB::c_in1;
    #endif

    binary_op_init_common(cb_inp0, cb_inp1);

    for(uint32_t block = 0; block < per_core_block_cnt; ++block) {

        cb_reserve_back(tt::CB::c_out0, per_core_block_size);

        for(uint32_t t = 0; t < per_core_block_size; ++t)
        {
            #ifdef SFPU_OP_INIT_PRE_IN0_0
                ACQ();
                cb_wait_front(tt::CB::c_in0, 1);

                copy_tile_init(); // need to copy from CB to DST to be able to run sfpu math
                copy_tile(tt::CB::c_in0, 0, 0); // copy from c_in[0] to DST[0]
                cb_pop_front(tt::CB::c_in0, 1);

                cb_reserve_back(cb_inp0, 1);
                SFPU_OP_INIT_PRE_IN0_0
                SFPU_OP_FUNC_PRE_IN0_0
                pack_tile(0, cb_inp0); // DST[0]->cb
                cb_push_back(cb_inp0, 1);
                REL();
            #endif
            #ifdef SFPU_OP_INIT_PRE_IN1_0
                ACQ();
                cb_wait_front(tt::CB::c_in1, 1);

                copy_tile_init(); // need to copy from CB to DST to be able to run sfpu math
                copy_tile(tt::CB::c_in1, 0, 0); // copy from c_in[0] to DST[0]
                cb_pop_front(tt::CB::c_in1, 1);

                cb_reserve_back(cb_inp1, 1);
                SFPU_OP_INIT_PRE_IN1_0
                SFPU_OP_FUNC_PRE_IN1_0
                pack_tile(0, cb_inp1); // DST[0]->cb
                cb_push_back(cb_inp1, 1);
                REL();
            #endif
            ACQ();
            cb_wait_front(cb_inp0, 1);
            cb_wait_front(cb_inp1, 1);
            #if ELTWISE_OP_CODE == 0
                add_tiles_init();
            #elif ELTWISE_OP_CODE == 1
                sub_tiles_init();
            #else
                mul_tiles_init();
            #endif

            ELTWISE_OP(cb_inp0, cb_inp1, 0, 0, 0);

            #ifdef SFPU_OP_INIT_0
            SFPU_OP_INIT_0
            SFPU_OP_FUNC_0
            #endif

            #ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
            #endif

            pack_tile(0, tt::CB::c_out0);
            cb_pop_front(cb_inp0, 1);
            cb_pop_front(cb_inp1, 1);
            REL();
        }
        cb_push_back(tt::CB::c_out0, per_core_block_size);
    }

}
}
