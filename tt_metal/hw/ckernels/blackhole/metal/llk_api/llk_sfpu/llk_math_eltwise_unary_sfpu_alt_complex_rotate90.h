// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_alt_complex_rotate90.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_alt_complex_rotate90_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::alt_complex_rotate90, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_alt_complex_rotate90(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_alt_complex_rotate90<APPROXIMATE>, dst_index, vector_mode);
}

}  // namespace ckernel
