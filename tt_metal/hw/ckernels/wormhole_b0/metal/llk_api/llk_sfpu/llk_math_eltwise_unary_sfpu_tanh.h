// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_tanh.h"

namespace ckernel {

template <bool APPROXIMATE, bool fast_and_approx = true>
inline void llk_math_eltwise_unary_sfpu_tanh_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::tanh, APPROXIMATE>(sfpu::tanh_init<APPROXIMATE, fast_and_approx>);
}

template <bool APPROXIMATE, bool fast_and_approx = true, bool fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_tanh(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_tanh<APPROXIMATE, fast_and_approx, fp32_dest_acc_en, ITERATIONS>,
        dst_index,
        vector_mode);
}

}  // namespace ckernel
