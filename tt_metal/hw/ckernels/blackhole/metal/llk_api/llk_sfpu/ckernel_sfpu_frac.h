// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel::sfpu {

// frac(x) = x - trunc(x)
// Uses IEEE 754 bit manipulation to extract the fractional part:
//   Case 1: E < 0 (|x| < 1) => result = x (entire value is fractional)
//   Case 2: E >= 23 (x is exact integer) => result = 0
//   Case 3: 0 <= E < 23 (mixed) => mask mantissa bits, subtract trunc(x) from x
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_frac() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];

        // Get the debiased exponent E of v
        // For IEEE 754: value = (-1)^s * 2^E * (1 + mantissa)
        // exexp returns the debiased exponent (biased_exp - 127)
        sfpi::vInt exp = sfpi::exexp(v);

        // Default result is 0 (for the E >= 23 case: exact integer)
        sfpi::vFloat result = 0.0f;

        // Case 1: E < 0 means |x| < 1, the entire value is fractional
        v_if(exp < 0) { result = v; }
        v_endif;

        // Case 3: 0 <= E < 23 means mixed integer+fractional parts
        // Create a mask to zero out the fractional mantissa bits
        // The mantissa has 23 bits. If E bits are integer part,
        // then (23 - E) bits are fractional. We need to mask those out.
        // mask = 0xFFFFFFFF << (23 - E) = ~0 << (23 - E)
        v_if(exp >= 0) {
            sfpi::vInt shift_amt = sfpi::vInt(23) - exp;

            v_if(shift_amt > 0) {
                // Build mask: all 1s shifted left by (23 - E) bits
                // This zeros out the fractional mantissa bits
                sfpi::vUInt mask = sfpi::shft(sfpi::vUInt(0xFFFFFFFF), shift_amt);

                // Apply mask to get trunc(x): zero out fractional bits
                sfpi::vInt v_bits = sfpi::reinterpret<sfpi::vInt>(v);
                sfpi::vFloat trunc_val = sfpi::reinterpret<sfpi::vFloat>(v_bits & sfpi::reinterpret<sfpi::vInt>(mask));

                // frac(x) = x - trunc(x)
                result = v - trunc_val;
            }
            v_endif;
        }
        v_endif;

        // Case 2 (E >= 23) is already handled: result stays 0.0f

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
