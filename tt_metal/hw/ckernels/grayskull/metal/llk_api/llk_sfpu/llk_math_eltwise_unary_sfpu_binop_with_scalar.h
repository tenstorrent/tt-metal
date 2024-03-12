// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_binop_with_unary.h"
#include "llk_math_eltwise_unary_sfpu_1_param.h"
#include "llk_math_eltwise_unary_sfpu_common_includes.h"
#include "llk_math_eltwise_unary_sfpu_init.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE, int binop_mode, DstSync Dst = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_binop_with_scalar(uint dst_index, uint32_t param1, int vector_mode = VectorMode::RC ) {
    llk_math_eltwise_unary_sfpu_1_param<APPROXIMATE, Dst>(
        ckernel::sfpu::calculate_binop_with_scalar<APPROXIMATE, binop_mode, 4>,
        ckernel::sfpu::calculate_binop_with_scalar<APPROXIMATE, binop_mode, 4>,
        dst_index,
        vector_mode,
        param1);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_binop_with_scalar_init() {
    llk_math_eltwise_unary_sfpu_init<APPROXIMATE>();
}

}  // namespace ckernel
