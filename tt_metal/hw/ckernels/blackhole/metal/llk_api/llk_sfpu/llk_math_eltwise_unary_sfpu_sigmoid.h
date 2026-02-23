// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_sigmoid.h"

namespace ckernel {

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_unary_sfpu_sigmoid_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::sigmoid, APPROX_MODE>(sfpu::sigmoid_init<APPROX_MODE>);
}

template <ckernel::ApproximationMode APPROX_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_sigmoid(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(
        sfpu::calculate_sigmoid<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>, dst_index, vector_mode);
}

}  // namespace ckernel
