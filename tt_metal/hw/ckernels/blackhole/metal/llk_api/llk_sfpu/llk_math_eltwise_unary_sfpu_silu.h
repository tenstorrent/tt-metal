// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_silu.h"

namespace ckernel {

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_unary_sfpu_silu_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::silu, APPROX_MODE>(sfpu::silu_init<APPROX_MODE>);
}

template <ckernel::ApproximationMode APPROX_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_silu(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::calculate_silu<is_fp32_dest_acc_en, ITERATIONS>, dst_index, vector_mode);
}

}  // namespace ckernel
