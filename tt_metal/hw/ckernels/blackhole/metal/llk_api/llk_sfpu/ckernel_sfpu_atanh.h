// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_log.h"
#include "sfpu/ckernel_sfpu_recip.h"

namespace ckernel::sfpu {

// atanh(x) = 0.5 * ln((1 + x) / (1 - x))  for |x| < 1
// Implementation: compute (1+x), compute (1-x), compute reciprocal of (1-x),
// multiply to get ratio, compute ln, multiply by 0.5
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_atanh() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // Compute numerator: 1 + x
        sfpi::vFloat num = x + sfpi::vConst1;

        // Compute denominator: 1 - x
        sfpi::vFloat den = sfpi::vConst1 - x;

        // Compute reciprocal of denominator: 1 / (1 - x)
        sfpi::vFloat recip_den = _sfpu_reciprocal_<2>(den);

        // Compute ratio: (1 + x) / (1 - x)
        sfpi::vFloat ratio = num * recip_den;

        // Compute ln((1 + x) / (1 - x))
        sfpi::vFloat log_val = _calculate_log_body_no_init_(ratio);

        // Multiply by 0.5
        sfpi::dst_reg[0] = log_val * 0.5f;

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void atanh_init() {
    _init_sfpu_reciprocal_<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
