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
        TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::LREG4, ADDR_MOD_7, 0);
        TTI_SFPNOT(0, p_sfpu::LREG0, p_sfpu::LREG0, 1);
        TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::LREG4, ADDR_MOD_7, 0);
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
