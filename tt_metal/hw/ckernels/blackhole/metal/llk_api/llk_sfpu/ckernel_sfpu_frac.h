// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel::sfpu {

// frac(x) = x - trunc(x)
//
// Matches PyTorch torch.frac() semantics (truncation toward zero, not floor).
//
// Algorithm:
// 1. Extract unbiased exponent from x.
// 2. If exp < 0 (|x| < 1): trunc(x) = 0, so frac = x.
// 3. If exp >= 23: x is already an integer, frac = 0.
// 4. Otherwise (0 <= exp < 23): compute trunc(x) by masking out fractional bits.
// 5. Result = x - trunc(x).
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_frac() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // Default: frac = 0 for integers (exp >= 23)
        sfpi::vFloat trunc_x = x;

        // Extract unbiased exponent
        sfpi::vInt exp = sfpi::exexp(x);

        // Case 1: |x| < 1 (exp < 0) — trunc toward zero gives 0
        v_if(exp < 0) { trunc_x = 0.0f; }
        v_endif;

        // Case 2: 0 <= exp < 23 (has fractional bits in float32)
        v_if(exp >= 0 && exp < 23) {
            // Create bitmask to zero out fractional mantissa bits.
            // IEEE 754 float32 has 23 mantissa bits. For exponent e,
            // the lowest (23 - e) bits are fractional.
            // mask = 0xFFFFFFFF << (23 - exp)
            sfpi::vUInt shift = sfpi::vUInt(23 - exp);
            sfpi::vInt mask = sfpi::vInt(-1) << shift;

            // Apply mask to get trunc(x) (round toward zero)
            sfpi::vInt xi = sfpi::reinterpret<sfpi::vInt>(x);
            trunc_x = sfpi::reinterpret<sfpi::vFloat>(xi & mask);
        }
        v_endif;

        // frac(x) = x - trunc(x)
        sfpi::dst_reg[0] = x - trunc_x;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
