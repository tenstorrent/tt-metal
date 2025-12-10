// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_elu.h"
#include "llk_defs.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_elu_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::elu, APPROXIMATE>(sfpu::elu_init<(APPROXIMATE ? ApproximationMode::Fast : ApproximationMode::Precise)>);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_elu(uint dst_index, uint param0) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_elu<(APPROXIMATE ? ApproximationMode::Fast : ApproximationMode::Precise), is_fp32_dest_acc_en>, dst_index, (int)VectorMode::RC, param0);
}

}  // namespace ckernel
