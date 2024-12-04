// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "noc_nonblocking_api.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_bitwise_not() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vInt input = dst_reg[0];
        vInt res;

        v_if(input < 0) {
            vInt unsigned_mag = input & 0x7FFFFFFF;
            res = unsigned_mag - 1;
        }
        v_else {
            res = setsgn(input, -1);
            res = res + 1;
        }
        v_endif;

        dst_reg[0] = res;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
