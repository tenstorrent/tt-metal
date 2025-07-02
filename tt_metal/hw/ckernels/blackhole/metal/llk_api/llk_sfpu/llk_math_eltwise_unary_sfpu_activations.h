// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_init.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_hardshrink_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::hardshrink, APPROXIMATE>();
}

template <bool APPROXIMATE, ckernel::ActivationType Activation, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_hardshrink(
    uint dst_index, float param0, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        static_cast<void (*)(uint)>(ckernel::sfpu::_calculate_activation_<APPROXIMATE, Activation, ITERATIONS>),
        dst_index,
        vector_mode,
        param0);
}
}  // namespace ckernel
