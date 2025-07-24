// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "ckernel_sfpu_binary.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE, ckernel::BinaryOp BINOP>
inline void llk_math_eltwise_binary_sfpu_binop_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROXIMATE>(
        ckernel::sfpu::sfpu_binary_init<APPROXIMATE, BINOP>);
}

template <bool APPROXIMATE, ckernel::BinaryOp BINOP>
inline void llk_math_eltwise_binary_sfpu_binop(uint dst_index0, uint32_t dst_index1, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_sfpu_binary<APPROXIMATE, BINOP>, dst_index0, dst_index1, vector_mode);
}

}  // namespace ckernel
