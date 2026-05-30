// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "ckernel_sfpu_isclose.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_isclose_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::isclose>();
}

template <bool APPROXIMATE, bool EQUAL_NAN = false, int ITERATIONS = 8>
inline void llk_math_eltwise_binary_sfpu_isclose(
    uint32_t dst_index0,
    uint32_t dst_index1,
    uint32_t odst,
    uint32_t rtol_bits,
    uint32_t atol_bits,
    VectorMode vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_(
        ckernel::sfpu::calculate_sfpu_isclose<APPROXIMATE, ITERATIONS, EQUAL_NAN>,
        dst_index0,
        dst_index1,
        odst,
        vector_mode,
        rtol_bits,
        atol_bits);
}

}  // namespace ckernel
