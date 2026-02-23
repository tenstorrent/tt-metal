// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_exp2.h"

namespace ckernel {

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_unary_sfpu_exp2_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::exp2, APPROX_MODE>(sfpu::exp2_init<APPROX_MODE>);
}

template <ckernel::ApproximationMode APPROX_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_exp2(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::calculate_exp2<APPROX_MODE, is_fp32_dest_acc_en>, dst_index, vector_mode);
}

}  // namespace ckernel
