// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel::sfpu {

// frac(x) = x - trunc(x)
// Returns the fractional part of x, preserving the sign of x.
// Matches PyTorch torch.frac() semantics.
//
// Algorithm:
//   1. Extract the debiased exponent E from x.
//   2. If E >= 23: x is an exact integer, frac = 0.
//   3. If E < 0:   |x| < 1, so trunc(x) = 0, frac = x.
//   4. Otherwise:  mask off the lower (23 - E) mantissa bits to get trunc(x),
//                  then frac = x - trunc(x).
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_frac() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];

        // Get the debiased exponent: for value = sign * 2^E * 1.mantissa
        sfpi::vInt exp = sfpi::exexp(v);

        // Default: result = v (handles E < 0 case: entire value is fractional)
        sfpi::vFloat result = v;

        // Case: E >= 23 => no fractional bits, result = 0
        v_if(exp >= 23) { result = 0.0f; }
        v_endif;

        // Case: 0 <= E < 23 => mixed integer + fractional
        v_if(exp >= 0) {
            v_if(exp < 23) {
                // Zero the lower (23 - E) mantissa bits to get trunc(x).
                // mask = all_ones << (23 - E)  zeroes the fractional mantissa bits.
                sfpi::vInt shift_amt = 23 - exp;
                sfpi::vUInt all_ones = sfpi::vUInt(0xFFFFFFFF);
                sfpi::vUInt mask = sfpi::shft(all_ones, sfpi::reinterpret<sfpi::vInt>(shift_amt));

                sfpi::vInt int_bits = sfpi::reinterpret<sfpi::vInt>(v);
                sfpi::vInt trunc_bits = int_bits & sfpi::reinterpret<sfpi::vInt>(mask);
                sfpi::vFloat trunc_val = sfpi::reinterpret<sfpi::vFloat>(trunc_bits);

                result = v - trunc_val;
            }
            v_endif;
        }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
