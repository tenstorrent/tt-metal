// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <int ITERATIONS = 8>
inline void calculate_binary_max(const uint dst_offset) {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr uint dst_tile_size = 64;

        TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, 0);                          // a
        TT_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_3, dst_offset * dst_tile_size);  // b

        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, 1);
        TTI_SFPNOP;

        TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, 0);
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
