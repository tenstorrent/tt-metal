/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "llk_math_eltwise_unary_sfpu_0_param.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_logical_not_unary_op(uint dst_index, int vector_mode) {
    llk_math_eltwise_unary_sfpu<SfpuType::logical_not_unary, APPROXIMATE, dst_sync>(dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_logical_not_unary_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::logical_not_unary, APPROXIMATE>();
}

void logical_not_unary_init() {
ckernel::sfpu::llk_math_eltwise_unary_sfpu_logical_not_unary_init<false>();
}

template <bool APPROXIMATION_MODE>
inline void calculate_logical_not_unary()
{
    #pragma GCC unroll 0
    for (int d = 0; d < WHB0_ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        v_if (v == 0) {
            dst_reg[0] = 1.0f;
        }v_else {
            dst_reg[0] = 0.0f;
        }
        v_endif;
        dst_reg++;
    }
}

template <bool APPROXIMATE, DstSync Dst = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_logical_not_unary_op(uint dst_index) {
    llk_math_eltwise_unary_sfpu_0_param<APPROXIMATE, Dst>
                                (ckernel::sfpu::calculate_logical_not_unary<APPROXIMATE>,
				 ckernel::sfpu::calculate_logical_not_unary<APPROXIMATE>,
				 dst_index, Dim::RC);
}

}  // namespace sfpu
}  // namespace ckernel
