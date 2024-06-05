// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_binop_with_unary.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_init.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE, int binop_mode>
inline void llk_math_eltwise_unary_sfpu_binop_with_scalar(uint dst_index, uint32_t param1, int vector_mode = VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_binop_with_scalar<APPROXIMATE, binop_mode, 8>,
        dst_index,
        vector_mode,
        param1);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_binop_with_scalar_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROXIMATE>();
}

}  // namespace ckernel
