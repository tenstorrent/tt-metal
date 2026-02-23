// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_add_rsqrt.h"

namespace ckernel {

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_unary_sfpu_add_rsqrt_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::rsqrt, APPROX_MODE>(sfpu::init_add_rsqrt<APPROX_MODE>);
}

template <ckernel::ApproximationMode APPROX_MODE, bool fp32_dest_acc_en, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_add_rsqrt(
    uint dst_index, uint32_t param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::calculate_add_rsqrt<APPROX_MODE, ITERATIONS, fp32_dest_acc_en>, dst_index, vector_mode, param0);
}

}  // namespace ckernel
