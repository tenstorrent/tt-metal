// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_add_rsqrt.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_add_rsqrt_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::rsqrt, APPROXIMATE>(sfpu::init_add_rsqrt<APPROXIMATE>);
}

template <bool APPROXIMATE, bool fp32_dest_acc_en, bool FAST_APPROX, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_add_rsqrt(
    uint dst_index, uint32_t param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_add_rsqrt<APPROXIMATE, ITERATIONS, fp32_dest_acc_en, FAST_APPROX>,
        dst_index,
        vector_mode,
        param0);
}

}  // namespace ckernel
