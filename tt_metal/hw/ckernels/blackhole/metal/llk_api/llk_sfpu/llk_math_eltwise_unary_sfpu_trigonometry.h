// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_trigonometry.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

namespace ckernel {

// New LLK SFPU APIs

// sine
SFPU_TRIG_KERNEL(sine)

// cosine
SFPU_TRIG_KERNEL(cosine)

// tangent
SFPU_TRIG_KERNEL(tan)

// asin
SFPU_UNARY_KERNEL(asin)

// acos
SFPU_UNARY_KERNEL(acos)

// acosh
SFPU_INVERSE_HYPERBOLIC_KERNEL(acosh, 8)

// atan
SFPU_INIT_KERNEL(atan, sfpu::atan_init)

// asinh
SFPU_INVERSE_HYPERBOLIC_KERNEL(asinh, 8)

// atanh
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_atanh_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::atanh, APPROXIMATE>(ckernel::sfpu::_init_atanh_<APPROXIMATE>);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_atanh(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::_calculate_atanh_<APPROXIMATE, is_fp32_dest_acc_en, ITERATIONS>, dst_index, vector_mode);
}

}  // namespace ckernel
