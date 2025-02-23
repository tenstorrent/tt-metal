// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_interleaved_complex_rotate90.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_interleaved_complex_rotate90_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::interleaved_complex_rotate90, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_interleaved_complex_rotate90(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_interleaved_complex_rotate90<APPROXIMATE>, dst_index, vector_mode);
}

}  // namespace ckernel
