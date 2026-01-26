// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_mask() {
    const bool exponent_size_8 = true;
    const int mask_val_idx = 32;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat mask = dst_reg[mask_val_idx];
        v_if(_sfpu_is_fp16_zero_(mask, exponent_size_8)) { dst_reg[0] = vConst0; }
        v_endif;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_int_mask() {
    const int mask_idx = 32;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vInt mask = dst_reg[mask_idx];
        v_if(mask == 0) { dst_reg[0] = vConst0; }
        v_endif;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_mask_posinf() {
    const bool exponent_size_8 = true;
    const int mask_val_idx = 32;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat mask = dst_reg[mask_val_idx];
        v_if(_sfpu_is_fp16_zero_(mask, exponent_size_8)) { dst_reg[0] = std::numeric_limits<float>::infinity(); }
        v_endif;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
