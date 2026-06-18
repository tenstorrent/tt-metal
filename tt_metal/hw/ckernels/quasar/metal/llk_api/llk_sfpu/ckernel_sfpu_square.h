// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {
// Calculates SQUARE for number of rows of output SFPU ops (Quasar = 2 rows)
template <bool APPROXIMATION_MODE>
inline void _calculate_square_sfp_rows_() {
    sfpi::vFloat v = sfpi::dst_reg[0];  // load x from dest (SFPLOAD)
    sfpi::dst_reg[0] = v * v;           // x * x via SFPMUL, store back to dest (SFPSTORE)
}

template <int ITERATIONS = SFPU_ITERATIONS>
inline void _calculate_square_() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        _calculate_square_sfp_rows_<false>();
        sfpi::dst_reg++;  // advances dest counter by one SFP row pair (TTINCRWC)
    }
}

}  // namespace sfpu
}  // namespace ckernel
