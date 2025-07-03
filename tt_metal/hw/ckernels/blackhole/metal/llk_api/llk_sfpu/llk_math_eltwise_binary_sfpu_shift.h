// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "ckernel_sfpu_shift.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_shift_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROXIMATE>();
}

template <bool APPROXIMATE, InstrModLoadStore INSTRUCTION_MODE = INT32, bool SIGN_MAGNITUDE_FORMAT = false>
inline void llk_math_eltwise_binary_sfpu_left_shift(
    uint dst_index0, uint32_t dst_index1, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_binary_left_shift<APPROXIMATE, 8, INSTRUCTION_MODE, SIGN_MAGNITUDE_FORMAT>,
        dst_index0,
        dst_index1,
        vector_mode);
}

template <bool APPROXIMATE, InstrModLoadStore INSTRUCTION_MODE = INT32, bool SIGN_MAGNITUDE_FORMAT = false>
inline void llk_math_eltwise_binary_sfpu_right_shift(
    uint dst_index0, uint32_t dst_index1, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_binary_right_shift<APPROXIMATE, 8, INSTRUCTION_MODE, SIGN_MAGNITUDE_FORMAT>,
        dst_index0,
        dst_index1,
        vector_mode);
}

}  // namespace ckernel
