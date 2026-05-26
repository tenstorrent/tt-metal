// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_ternary_sfpu_params.h"
#include "ckernel_sfpu_snake_beta.h"

namespace ckernel {

template <bool APPROXIMATE, bool is_fp32_dest_acc_en, DataFormat data_format, int ITERATIONS = 8>
inline void llk_math_eltwise_ternary_sfpu_snake_beta(
    uint dst_index0, uint dst_index1, uint dst_index2, uint odst, VectorMode vector_mode = VectorMode::RC) {
    _llk_math_eltwise_ternary_sfpu_params_(
        sfpu::calculate_snake_beta<APPROXIMATE, is_fp32_dest_acc_en, data_format, ITERATIONS>,
        dst_index0,
        dst_index1,
        dst_index2,
        odst,
        vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_ternary_sfpu_snake_beta_init() {
    _llk_math_eltwise_ternary_sfpu_init_<SfpuType::snake_beta>();
    sfpu::snake_beta_init<APPROXIMATE>();  // calls _init_sfpu_reciprocal_<APPROXIMATE>()
}

}  // namespace ckernel
