// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_exp.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_rpow_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::rpow, APPROXIMATE>(
        ckernel::sfpu::_init_exponential_<APPROXIMATE, /*FAST_APPROX=*/APPROXIMATE, /*SCALE=*/p_sfpu::kCONST_1_FP16B>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_rpow(uint dst_index, uint32_t log_val, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::_calculate_rpow_<APPROXIMATE, 8>, dst_index, log_val, vector_mode);
}

}  // namespace ckernel
