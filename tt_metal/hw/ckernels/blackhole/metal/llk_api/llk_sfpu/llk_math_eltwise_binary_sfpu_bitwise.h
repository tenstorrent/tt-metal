// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "ckernel_sfpu_binary_bitwise.h"
#include "ckernel_sfpu_binary_bitwise_uint16.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_bitwise_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROXIMATE>();
}

template <bool APPROXIMATE, ckernel::sfpu::BinaryBitwiseOp BITWISE_OP>
inline void llk_math_eltwise_binary_sfpu_bitwise(
    uint dst_index0, uint32_t dst_index1, int vector_mode = VectorMode::RC) {
    llk_math_eltwise_binary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_sfpu_binary_bitwise<APPROXIMATE, BITWISE_OP>, dst_index0, dst_index1, vector_mode);
}

template <bool APPROXIMATE, ckernel::sfpu::BinaryBitwiseOp BITWISE_OP>
inline void llk_math_eltwise_binary_sfpu_bitwise_uint16(
    uint dst_index0, uint32_t dst_index1, int vector_mode = VectorMode::RC) {
    llk_math_eltwise_binary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_sfpu_binary_bitwise_uint16<APPROXIMATE, BITWISE_OP>,
        dst_index0,
        dst_index1,
        vector_mode);
}

}  // namespace ckernel
