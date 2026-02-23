// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"

namespace ckernel {

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_binary_sfpu_xlogy_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROX_MODE>(
        ckernel::sfpu::_sfpu_binary_init_<APPROX_MODE, BinaryOp::XLOGY>);
}

template <ckernel::ApproximationMode APPROX_MODE, int ITERATIONS = 8>
inline void llk_math_eltwise_binary_sfpu_xlogy(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::_calculate_sfpu_binary_<APPROX_MODE, BinaryOp::XLOGY, ITERATIONS>,
        dst_index0,
        dst_index1,
        odst,
        vector_mode);
}

}  // namespace ckernel
