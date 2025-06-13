// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_unary_selu.h"

namespace ckernel {

// New LLK SFPU APIs
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_selu_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::selu, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_selu(
    uint dst_index, uint param0, uint param1, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_selu<APPROXIMATE>, dst_index, vector_mode, param0, param1);
}
}  // namespace ckernel
