// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_recip.h"

namespace ckernel::sfpu {

// softsign(x) = x / (1 + |x|)
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softsign() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];

        // Compute denominator: 1 + |x|
        sfpi::vFloat denom = sfpi::abs(v) + sfpi::vConst1;

        // Compute reciprocal of denominator: 1 / (1 + |x|)
        sfpi::vFloat recip = _sfpu_reciprocal_<2>(denom);

        // Result: x * (1 / (1 + |x|))
        sfpi::dst_reg[0] = v * recip;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void softsign_init() {
    _init_sfpu_reciprocal_<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
