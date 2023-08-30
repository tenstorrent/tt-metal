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

namespace NAMESPACE {
void MAIN {
    binary_op_specific_init(ELTWISE_OP_CODE);
    binary_op_init_common(0, 1);

    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_size = get_compile_time_arg_val(1);

    for(uint32_t block = 0; block < per_core_block_cnt; ++block) {

        cb_reserve_back(tt::CB::c_out0, per_core_block_size);

        for(uint32_t t = 0; t < per_core_block_size; ++t)
        {
            acquire_dst(tt::DstMode::Half);

            cb_wait_front(tt::CB::c_in0, 1);
            cb_wait_front(tt::CB::c_in1, 1);

            // ELTWISE_OP is passed in via add_define
            ELTWISE_OP(tt::CB::c_in0, tt::CB::c_in1, 0, 0, 0);

            #ifdef SFPU_OP_INIT_0
            SFPU_OP_INIT_0
            SFPU_OP_FUNC_0
            #endif

            #ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
            #endif


            pack_tile(0, tt::CB::c_out0);

            cb_pop_front(tt::CB::c_in0, 1);
            cb_pop_front(tt::CB::c_in1, 1);

            release_dst(tt::DstMode::Half);
        }

        cb_push_back(tt::CB::c_out0, per_core_block_size);
    }

}
}
