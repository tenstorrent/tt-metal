// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_instr_params.h"
#include "ckernel_sfpu_rand_uint.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_rand_uint_init(uint seed = 0) {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROXIMATE>(sfpu::rand_uint_init<APPROXIMATE>, seed);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_rand_uint(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    // Init a nice per-lane counter
    TTI_SFPMOV(0, p_sfpu::LTILEID, p_sfpu::LREG0, 0);
    TTI_SFPSHFT(-1 & 0xfff, 0, p_sfpu::LREG0, 1);

    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(ckernel::sfpu::rand_uint<APPROXIMATE>, dst_index, vector_mode);
}

}  // namespace ckernel
