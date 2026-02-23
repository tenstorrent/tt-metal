// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_unary_selu.h"
namespace ckernel {

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_unary_sfpu_selu_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::selu, APPROX_MODE>();
}

template <ckernel::ApproximationMode APPROX_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_selu(
    uint dst_index, uint scale, uint alpha, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::calculate_selu<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>,
        dst_index,
        vector_mode,
        scale,
        alpha);
}

}  // namespace ckernel
