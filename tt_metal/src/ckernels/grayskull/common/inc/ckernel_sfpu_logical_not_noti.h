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

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_logical_not_unary_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::logical_not_unary, APPROXIMATE>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_logical_not_unary()
{
    #pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++)
    {
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
                                (ckernel::sfpu::calculate_logical_not_unary<APPROXIMATE,4>,
				 ckernel::sfpu::calculate_logical_not_unary<APPROXIMATE,4>,
				 dst_index, Dim::RC);
}

}  // namespace sfpu
}  // namespace ckernel
