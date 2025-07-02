// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "ckernel_sfpu_binary_max_min.h"

namespace ckernel {

// New LLK SFPU APIs

// Binary maximum
template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_binary_max_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROXIMATE>();
}

template <bool APPROXIMATE, InstrModLoadStore INSTRUCTION_MODE>
inline void llk_math_eltwise_binary_sfpu_binary_max(
    uint dst_index0, uint32_t dst_index1, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_binary_max_min<INSTRUCTION_MODE, true>, dst_index0, dst_index1, vector_mode);
}

// Binary minimum
template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_binary_min_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROXIMATE>();
}

template <bool APPROXIMATE, InstrModLoadStore INSTRUCTION_MODE>
inline void llk_math_eltwise_binary_sfpu_binary_min(
    uint dst_index0, uint32_t dst_index1, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_binary_max_min<INSTRUCTION_MODE, false>, dst_index0, dst_index1, vector_mode);
}

}  // namespace ckernel
