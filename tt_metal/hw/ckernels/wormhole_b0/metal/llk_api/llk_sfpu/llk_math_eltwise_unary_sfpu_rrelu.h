// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_rrelu.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_rrelu_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::rrelu, APPROXIMATE>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_rrelu(
    uint dst_index, int vector_mode = (int)VectorMode::RC, std::uint32_t slope = 0) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_rrelu<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, slope);
}

}  // namespace ckernel
