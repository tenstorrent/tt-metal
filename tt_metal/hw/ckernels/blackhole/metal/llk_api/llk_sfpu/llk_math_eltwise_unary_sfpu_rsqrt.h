// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_rsqrt.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_rsqrt_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::rsqrt, APPROXIMATE>(sfpu::rsqrt_init<APPROXIMATE>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_rsqrt(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    // APPROXIMATE = true -> approximate fast mode
    //               false -> high precision mode
    // The algorithm uses Newton's method based on no.of iteration better approximation can be calculated

    // if (APPROXIMATE) {
    //     _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
    //                         ckernel::sfpu::calculate_rsqrt<APPROXIMATE, 8, 10>,
    //                         dst_index, vector_mode);
    // } else {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_rsqrt<APPROXIMATE, 8, 25>, dst_index, vector_mode);
    // }
}

}  // namespace ckernel
