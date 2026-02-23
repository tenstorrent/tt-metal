// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "ckernel_sfpu_mul_int32.h"

namespace ckernel {

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_binary_sfpu_mul_int32_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::mul_int32, APPROX_MODE>(sfpu::mul_int32_init<APPROX_MODE>);
}

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_binary_sfpu_mul_int32(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROX_MODE>(
        sfpu::mul_int32<APPROX_MODE>, dst_index0, dst_index1, odst, vector_mode);
}

}  // namespace ckernel
