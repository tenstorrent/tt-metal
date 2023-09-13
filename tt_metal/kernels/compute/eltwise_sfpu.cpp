// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api.h"

#if SFPU_OP_ISINF_ISNAN_INCLUDE
#include "compute_kernel_api/eltwise_unary/isinf_isnan.h"
#endif

#if SFPU_OP_ERF_ERFC_INCLUDE
#include "compute_kernel_api/eltwise_unary/erf_erfc.h"
#endif

#if SFPU_OP_LOGICAL_NOT_NOTI_INCLUDE
#include "compute_kernel_api/eltwise_unary/logical_not_noti.h"
#endif

#if SFPU_OP_ERFINV_INCLUDE
#include "compute_kernel_api/eltwise_unary/erfinv.h"
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

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    init_sfpu(tt::CB::c_in0);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CB::c_out0, per_core_block_dim);
        for(uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            acquire_dst(tt::DstMode::Half);

            // Pop tile after tile, copy to DST and pack
            cb_wait_front(tt::CB::c_in0, 1);

            copy_tile(tt::CB::c_in0, 0, 0);

            #ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
            #endif

            pack_tile(0, tt::CB::c_out0);

            cb_pop_front(tt::CB::c_in0, 1);

            release_dst(tt::DstMode::Half);
        }
        cb_push_back(tt::CB::c_out0, per_core_block_dim);
    }
}
}
