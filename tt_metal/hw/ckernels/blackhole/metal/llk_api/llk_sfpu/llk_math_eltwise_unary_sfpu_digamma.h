// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_digamma.h"

namespace ckernel {

template <bool APPROXIMATE, bool fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_digamma_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::digamma, APPROXIMATE>(sfpu::digamma_init<APPROXIMATE, fp32_dest_acc_en>);
}

template <bool APPROXIMATE, bool fp32_dest_acc_en, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_digamma(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        sfpu::calculate_digamma<APPROXIMATE, fp32_dest_acc_en, ITERATIONS>, dst_index, vector_mode);
}

}  // namespace ckernel
