// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_hardtanh.h"

namespace ckernel {

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_unary_sfpu_hardtanh_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::hardtanh, APPROX_MODE>();
}

template <ckernel::ApproximationMode APPROX_MODE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_hardtanh(
    uint dst_index, uint param0, uint param1, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::calculate_hardtanh<APPROX_MODE, ITERATIONS>, dst_index, vector_mode, param0, param1);
}

}  // namespace ckernel
