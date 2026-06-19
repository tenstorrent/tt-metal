// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

inline void _init_square_() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_6, csr_read<CSR::TRISC_ID>());
}

// Calculates SQUARE for number of rows of output SFPU ops (Quasar = 2 rows)
inline void _calculate_square_sfp_rows_() {
    sfpi::vFloat v = sfpi::dst_reg[0];  // load x from dest (SFPLOAD)
    sfpi::dst_reg[0].mode<>(ckernel::ADDR_MOD_6) = v * v;  // x * x via SFPMUL, store back to dest (SFPSTORE)
}

template <int ITERATIONS = SFPU_ITERATIONS>
inline void _calculate_square_() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        _calculate_square_sfp_rows_();
    }
}

}  // namespace sfpu
}  // namespace ckernel
