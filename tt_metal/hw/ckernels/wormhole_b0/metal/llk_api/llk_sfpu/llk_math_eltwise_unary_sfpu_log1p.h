// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_log1p.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_log1p_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::log1p, APPROXIMATE>(sfpu::log1p_init<APPROXIMATE>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_log1p(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_log1p<APPROXIMATE>, dst_index, vector_mode);
}

}  // namespace ckernel
