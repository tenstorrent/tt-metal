// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api.h"

#if SFPU_OP_ERF_ERFC_INCLUDE
#include "compute_kernel_api/eltwise_unary/erf_erfc.h"
#endif

#if SFPU_OP_EXP_INCLUDE
#include "compute_kernel_api/eltwise_unary/exp.h"
#endif

#if SFPU_OP_GELU_INCLUDE
#include "compute_kernel_api/eltwise_unary/gelu.h"
#endif

#if SFPU_OP_SQRT_INCLUDE
#include "compute_kernel_api/eltwise_unary/sqrt.h"
#endif

#if SFPU_OP_RECIP_INCLUDE
#include "compute_kernel_api/eltwise_unary/recip.h"
#endif

#if SFPU_OP_RELU_FAMILY_INCLUDE
#include "compute_kernel_api/eltwise_unary/relu.h"
#endif

#if SFPU_OP_ELU_INCLUDE
#include "compute_kernel_api/eltwise_unary/elu.h"
#endif

//#include "debug_print.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api.h"
#include "debug_print.h"

ALWI void ACQ() { acquire_dst(tt::DstMode::Half); }
ALWI void REL() { release_dst(tt::DstMode::Half); }

namespace NAMESPACE {
void MAIN {

    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_size = get_compile_time_arg_val(1);
    constexpr auto cb_ex = tt::CB::c_intermed0;
    constexpr auto cb_ex2 = tt::CB::c_intermed1;
    #ifdef SFPU_OP_PRE_INIT_0
        binary_op_init_common(cb_ex, cb_ex2);
    #endif
    #ifndef SFPU_OP_PRE_INIT_0
        binary_op_specific_init(ELTWISE_OP_CODE);
        binary_op_init_common(0, 1);
    #endif
    for(uint32_t block = 0; block < per_core_block_cnt; ++block) {

        cb_reserve_back(tt::CB::c_out0, per_core_block_size);

        for(uint32_t t = 0; t < per_core_block_size; ++t)
        {
            acquire_dst(tt::DstMode::Half);

            // start of prescaling
            #ifdef SFPU_OP_PRE_INIT_0
                cb_wait_front(tt::CB::c_in1, 1);

                copy_tile_init(); // need to copy from CB to DST to be able to run sfpu math
                copy_tile(tt::CB::c_in1, 0, 0); // copy from c_in[0] to DST[0]
                cb_pop_front(tt::CB::c_in1, 1);

                cb_reserve_back(cb_ex2, 1);
                exp_tile_init();
                exp_tile(0); // exp on DST[0]
                pack_tile(0, cb_ex2); // DST[0]->cb
                cb_push_back(cb_ex2, 1);
                REL();

                ACQ();
                cb_wait_front(tt::CB::c_in0, 1);
                copy_tile_init(); // need to copy from CB to DST to be able to run sfpu math
                copy_tile(tt::CB::c_in0, 0, 0); // copy from c_in[0] to DST[0]
                cb_pop_front(tt::CB::c_in0, 1);

                cb_reserve_back(cb_ex, 1);
                exp_tile_init();
                exp_tile(0); // exp on DST[0]
                pack_tile(0, cb_ex); // DST[0]->cb
                cb_push_back(cb_ex, 1);
                REL();

                ACQ();
                cb_wait_front(cb_ex, 1);
                cb_wait_front(cb_ex2, 1);
                #if ELTWISE_OP_CODE == 0
                    add_tiles_init();
                #elif ELTWISE_OP_CODE == 1
                    sub_tiles_init();
                #else
                    mul_tiles_init();
                #endif
                ELTWISE_OP(cb_ex, cb_ex2, 0, 0, 0);
            #endif
            //end of prescaling

           #ifndef SFPU_OP_PRE_INIT_0
                cb_wait_front(tt::CB::c_in0, 1);
                cb_wait_front(tt::CB::c_in1, 1);
                // ELTWISE_OP is passed in via add_define
                ELTWISE_OP(tt::CB::c_in0, tt::CB::c_in1, 0, 0, 0);
            #endif

            #ifdef SFPU_OP_INIT_0
            SFPU_OP_INIT_0
            SFPU_OP_FUNC_0
            #endif

            #ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
            #endif

            pack_tile(0, tt::CB::c_out0);
            #ifndef SFPU_OP_PRE_INIT_0
                cb_pop_front(tt::CB::c_in0, 1);
                cb_pop_front(tt::CB::c_in1, 1);
            #endif
            #ifdef SFPU_OP_PRE_INIT_0
                cb_pop_front(cb_ex2, 1);
                cb_pop_front(cb_ex, 1);
            #endif
            release_dst(tt::DstMode::Half);
        }

        cb_push_back(tt::CB::c_out0, per_core_block_size);
    }

}
}
