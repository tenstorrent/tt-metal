// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "llk_defs.h"
using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <ApproximationMode APPROX_MODE, int ITERATIONS = 8>
inline void calculate_left_shift(const uint shift_amt) {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(0,12,ADDR_MOD_7,0);
        TT_SFPSHFT(shift_amt,0,0,1);
        TTI_SFPSTORE(0,12,ADDR_MOD_7,0);
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
