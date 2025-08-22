// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_sqrt.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE, bool fp32_dest_acc_en, bool legacy_compat>
inline void llk_math_eltwise_unary_sfpu_sqrt(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_sqrt<APPROXIMATE, 8, fp32_dest_acc_en, legacy_compat>, dst_index, vector_mode);
}

template <bool APPROXIMATE, bool legacy_compat>
inline void llk_math_eltwise_unary_sfpu_sqrt_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::sqrt, APPROXIMATE>(sfpu::sqrt_init<APPROXIMATE, legacy_compat>);
}

}  // namespace ckernel
