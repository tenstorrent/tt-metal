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
inline void calculate_tiled_prod() {
    vFloat result = 1.0f;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        result *= v;
        dst_reg[0] = result;
        dst_reg++;
    }
    vFloat v = dst_reg[0];
    result *= v;
    dst_reg[0] = result;
    dst_reg++;
}

}  // namespace sfpu
}  // namespace ckernel
