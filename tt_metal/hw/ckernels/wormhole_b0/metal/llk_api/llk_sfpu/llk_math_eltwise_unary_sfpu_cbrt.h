// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_cbrt.h"

namespace ckernel {

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_unary_sfpu_cbrt_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::cbrt, APPROX_MODE>(sfpu::cube_root_init<APPROX_MODE>);
}

template <ckernel::ApproximationMode APPROX_MODE, bool fp32_dest_acc_en, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_cbrt(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(
        sfpu::calculate_cube_root<APPROX_MODE, fp32_dest_acc_en, ITERATIONS>, dst_index, vector_mode);
}

}  // namespace ckernel
