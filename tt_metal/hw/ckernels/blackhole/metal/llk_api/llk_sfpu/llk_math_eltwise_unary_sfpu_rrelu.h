// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_rrelu.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_rrelu_init(uint32_t seed = 0) {
    llk_math_eltwise_unary_sfpu_init<SfpuType::rrelu, APPROXIMATE>();
    ckernel::sfpu::_init_rrelu_(seed);
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_rrelu(
    uint dst_index, uint32_t lower, uint32_t range, uint32_t seed, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::_calculate_rrelu_<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, lower, range, seed);
}

}  // namespace ckernel
