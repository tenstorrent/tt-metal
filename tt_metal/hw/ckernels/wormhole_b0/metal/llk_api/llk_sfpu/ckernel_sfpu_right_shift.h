// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_right_shift(const uint shift_amt) {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vInt input = sfpi::dst_reg[0];
        sfpi::vUInt val = reinterpret<sfpi::vUInt>(input);

        v_if(input < 0) { val = setsgn(val - 1, 0); }
        v_endif;
        sfpi::vInt res = reinterpret<sfpi::vInt>(val >> shift_amt);

        v_if(input < 0) { res = setsgn(res + 1, input); }
        v_endif;

        sfpi::dst_reg[0] = res;
        sfpi::dst_reg++;
    }
}
}  // namespace sfpu
}  // namespace ckernel
