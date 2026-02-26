// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_lgamma.h"

namespace ckernel {

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_lgamma_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::lgamma, APPROXIMATE>(
        sfpu::lgamma_init<APPROXIMATE, is_fp32_dest_acc_en>);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_lgamma(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_lgamma_part_positive<APPROXIMATE, is_fp32_dest_acc_en>, dst_index, vector_mode);
}

}  // namespace ckernel
