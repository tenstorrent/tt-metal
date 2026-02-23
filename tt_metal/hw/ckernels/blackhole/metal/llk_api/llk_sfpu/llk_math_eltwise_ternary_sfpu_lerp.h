// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_ternary_sfpu_params.h"
#include "ckernel_sfpu_lerp.h"

namespace ckernel {

template <ckernel::ApproximationMode APPROX_MODE, bool is_fp32_dest_acc_en, DataFormat data_format, int ITERATIONS = 8>
inline void llk_math_eltwise_ternary_sfpu_lerp(
    uint dst_index0, uint dst_index1, uint dst_index2, uint odst, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_ternary_sfpu_params_<APPROX_MODE>(
        sfpu::calculate_lerp<APPROX_MODE, is_fp32_dest_acc_en, data_format, ITERATIONS>,
        dst_index0,
        dst_index1,
        dst_index2,
        odst,
        vector_mode);
}

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_ternary_sfpu_lerp_init() {
    _llk_math_eltwise_ternary_sfpu_init_<SfpuType::lerp>();
}

}  // namespace ckernel
