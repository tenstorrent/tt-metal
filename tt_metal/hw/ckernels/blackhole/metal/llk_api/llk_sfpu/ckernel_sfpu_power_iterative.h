// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_power_iterative(const uint exponent) {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat in = dst_reg[0];
        vFloat result = 1.0f;
        for (uint i = 0; i < exponent; i++) {
            result *= in;
        }
        dst_reg[0] = result;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
