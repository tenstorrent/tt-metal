// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_left_shift(const uint shift_amt) {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vInt val = dst_reg[0];
        vInt v = val;

        val = val << shift_amt;
        val = setsgn(val, v);
        dst_reg[0] = val;

        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
