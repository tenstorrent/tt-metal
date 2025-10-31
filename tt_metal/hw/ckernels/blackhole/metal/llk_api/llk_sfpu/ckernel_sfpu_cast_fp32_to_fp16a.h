// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void cast_fp32_to_fp16a() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // sfpi::vFloat val = sfpi::dst_reg[0];
        // sfpi::dst_reg[0] = sfpi::float_to_fp16a(val, 0);
        TTI_SFPLOAD(0, 0, 3, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 8);
        TTI_SFPSTORE(0, 1, 3, 0);
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
