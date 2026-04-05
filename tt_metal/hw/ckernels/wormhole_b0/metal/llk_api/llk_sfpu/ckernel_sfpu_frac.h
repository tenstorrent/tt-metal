// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel::sfpu {

// frac(x) = x - floor(x)
// Returns the fractional part of x, always in [0, 1).
//
// Algorithm:
//   1. Compute trunc(x) via IEEE754 mantissa bit masking.
//   2. For x >= 0: frac = x - trunc(x)
//   3. For x <  0: frac = x - trunc(x) + 1  (if x is not integer)
//      because floor(x) = trunc(x) - 1 for non-integer negative x.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_frac() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];

        // Get the debiased exponent: for 2^E * 1.mantissa, E tells us how
        // many integer bits exist above the radix point.
        sfpi::vInt exp = sfpi::exexp(v);

        // Start with frac = 0 (default for integers with E >= 23)
        sfpi::vFloat result = 0.0f;

        // Case: E < 0  =>  |x| < 1, entire value is fractional
        // trunc(x) = 0, so frac = x for positive, x + 1 for negative
        v_if(exp < 0) {
            result = v;
            v_if(v < 0.0f) { result = v + sfpi::vConst1; }
            v_endif;
        }
        v_endif;

        // Case: 0 <= E < 23  =>  mixed integer+fractional
        // Zero out the lower (23 - E) mantissa bits to get trunc(x),
        // then frac = x - trunc(x), adjusted for sign.
        v_if(exp >= 0) {
            v_if(exp < 23) {
                // Build a mask to zero the fractional mantissa bits.
                // We need mask = 0xFFFFFFFF << (23 - E).
                // In SFPI: start with all-ones, shift left by (23 - E).
                sfpi::vInt shift_amt = 23 - exp;
                sfpi::vUInt all_ones = sfpi::vUInt(0xFFFFFFFF);
                sfpi::vUInt mask = sfpi::shft(all_ones, sfpi::reinterpret<sfpi::vInt>(shift_amt));

                // Apply mask to get trunc bits (zeroing fractional mantissa bits).
                sfpi::vInt int_bits = sfpi::reinterpret<sfpi::vInt>(v);
                sfpi::vInt trunc_bits = int_bits & sfpi::reinterpret<sfpi::vInt>(mask);
                sfpi::vFloat trunc_val = sfpi::reinterpret<sfpi::vFloat>(trunc_bits);

                // Compute the difference
                sfpi::vFloat diff = v - trunc_val;

                // For positive x: frac = diff (which is >= 0)
                // For negative x: diff <= 0, so frac = diff + 1 (unless diff == 0)
                result = diff;
                v_if(v < 0.0f) {
                    v_if(diff < 0.0f) { result = diff + sfpi::vConst1; }
                    v_endif;
                }
                v_endif;
            }
            v_endif;
        }
        v_endif;
        // Case: E >= 23 is already handled by result = 0.0f default

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
