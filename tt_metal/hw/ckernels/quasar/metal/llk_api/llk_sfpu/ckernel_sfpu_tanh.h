// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_trisc_common.h"
#include "cmath_common.h"

namespace ckernel {
namespace sfpu {

// Calculates tanh for number of rows of output SFPU ops (Quasar = 2 rows)
inline void _calculate_tanh_sfp_rows_() {
    TTI_SFPLOAD(
        p_sfpu::LREG0,
        p_sfpu::sfpmem::DEFAULT,
        ADDR_MOD_7,
        0,
        0);  // load from dest into lreg[0], uses ADDR_MOD_7 (set to all zeroes)
    TTI_SFPNONLINEAR(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpnonlinear::TANH_MODE);  // tanh via SFPU nonlinear unit
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, 0, 0);                           // store from lreg[1] into dest register
}

template <int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_tanh() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        _calculate_tanh_sfp_rows_();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>();  // does the dest_reg++ (increments by
                                                                                   // 2 rows)
    }
}

}  // namespace sfpu
}  // namespace ckernel
