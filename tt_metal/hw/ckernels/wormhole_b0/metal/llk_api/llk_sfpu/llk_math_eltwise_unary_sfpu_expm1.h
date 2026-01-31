// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_expm1.h"
#include "llk_defs.h"

namespace ckernel {

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_expm1_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::expm1, APPROXIMATE>(sfpu::expm1_init<(APPROXIMATE ? ApproximationMode::Fast : ApproximationMode::Precise), is_fp32_dest_acc_en>);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_expm1(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_expm1<(APPROXIMATE ? ApproximationMode::Fast : ApproximationMode::Precise), is_fp32_dest_acc_en, 8>, dst_index, vector_mode);
}

}  // namespace ckernel
