// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_rsqrt.h"

namespace ckernel {

template <ckernel::ApproximationMode APPROX_MODE, bool legacy_compat>
inline void llk_math_eltwise_unary_sfpu_rsqrt_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::rsqrt, APPROX_MODE>(sfpu::rsqrt_init<APPROX_MODE, legacy_compat>);
}

template <ckernel::ApproximationMode APPROX_MODE, bool fp32_dest_acc_en, bool legacy_compat>
inline void llk_math_eltwise_unary_sfpu_rsqrt(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::calculate_rsqrt<APPROX_MODE, 8, fp32_dest_acc_en, legacy_compat>, dst_index, vector_mode);
}

}  // namespace ckernel
