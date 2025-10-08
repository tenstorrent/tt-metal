// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_bitwise_and(const uint value) {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vInt input = sfpi::dst_reg[0];
        sfpi::vInt res = input & value;
        sfpi::dst_reg[0] = res;
        sfpi::dst_reg++;
    }
}
}  // namespace sfpu
}  // namespace ckernel
