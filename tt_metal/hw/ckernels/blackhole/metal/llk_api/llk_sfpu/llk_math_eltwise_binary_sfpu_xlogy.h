// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_xlogy_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROXIMATE>(
        ckernel::sfpu::_sfpu_binary_init_<APPROXIMATE, BinaryOp::XLOGY>);
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_binary_sfpu_xlogy(uint dst_index0, uint32_t dst_index1, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::_calculate_sfpu_binary_<APPROXIMATE, BinaryOp::XLOGY, ITERATIONS>,
        dst_index0,
        dst_index1,
        vector_mode);
}

}  // namespace ckernel
