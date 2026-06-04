// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "ckernel_sfpu_binary.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_xlogy_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused>(ckernel::sfpu::sfpu_binary_init<APPROXIMATE, BinaryOp::XLOGY>);
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_binary_sfpu_xlogy(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, VectorMode vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_(
        ckernel::sfpu::calculate_sfpu_binary<APPROXIMATE, BinaryOp::XLOGY, ITERATIONS>,
        dst_index0,
        dst_index1,
        odst,
        vector_mode);
}

}  // namespace ckernel
