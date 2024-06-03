// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_erf_erfc.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_erf_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::erf, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_erfc_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::erfc, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_erf(uint dst_index, int param0 = 0) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_sfpu_erf_erfc<SfpuType::erf, APPROXIMATE>,
        dst_index,
        (int)VectorMode::RC);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_erfc(uint dst_index, int param0 = 0) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_sfpu_erf_erfc<SfpuType::erfc, APPROXIMATE>,
        dst_index,
        (int)VectorMode::RC);
}

}
