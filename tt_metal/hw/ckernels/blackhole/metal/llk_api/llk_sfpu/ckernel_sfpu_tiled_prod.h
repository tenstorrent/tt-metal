// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_tiled_prod() {
    sfpi::vFloat result = 1.0f;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        result *= v;
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
    sfpi::vFloat v = sfpi::dst_reg[0];
    result *= v;
    sfpi::dst_reg[0] = result;
    sfpi::dst_reg++;
}

}  // namespace sfpu
}  // namespace ckernel
