// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"
#include "ckernel_sfpu_mul_int32.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_square() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat in = dst_reg[0];
        vFloat result = in * in;

        dst_reg[0] = result;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_square_int32() {
    constexpr uint dst_offset = 0;
    mul_int32<APPROXIMATION_MODE, ITERATIONS>(dst_offset);
}

}  // namespace ckernel::sfpu
