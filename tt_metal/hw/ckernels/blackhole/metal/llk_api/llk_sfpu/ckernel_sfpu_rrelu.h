// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// rrelu_eval(x) = x if x >= 0, slope * x if x < 0
// where slope = (lower + upper) / 2 (precomputed on host)
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_rrelu_eval(uint32_t slope_u32) {
    sfpi::vFloat slope = sfpi::reinterpret<sfpi::vFloat>(sfpi::vInt(slope_u32));

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat result = x;

        v_if(x < 0.0f) { result = x * slope; }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// rrelu_train(x) = x if x >= 0, a * x if x < 0
// where a ~ Uniform(lower, upper) per element
//
// PRNG: We use the float bit-pattern of x as a per-element entropy source.
// We apply a multiplicative hash (Knuth golden ratio) to the reinterpreted
// integer bits, extract the lower mantissa bits to form a float in [1, 2),
// then map to [lower, upper].  This gives uncorrelated slopes for distinct
// input values.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_rrelu_train(uint32_t lower_u32, uint32_t upper_u32) {
    sfpi::vFloat lower = sfpi::reinterpret<sfpi::vFloat>(sfpi::vInt(lower_u32));
    sfpi::vFloat upper = sfpi::reinterpret<sfpi::vFloat>(sfpi::vInt(upper_u32));
    sfpi::vFloat range = upper - lower;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat result = x;

        v_if(x < 0.0f) {
            // Generate pseudo-random value in [0, 1) from input bits
            sfpi::vUInt bits = sfpi::reinterpret<sfpi::vUInt>(x);
            // Multiplicative hash with golden ratio constant
            bits = bits ^ (bits >> 7);
            bits = bits ^ (bits << 13);
            bits = bits ^ (bits >> 17);
            // Extract lower 23 bits (mantissa), set exponent to 127 -> [1.0, 2.0)
            sfpi::vUInt mantissa = bits & sfpi::vUInt(0x007FFFFF);
            sfpi::vUInt one_to_two = mantissa | sfpi::vUInt(0x3F800000);
            // Convert to [0, 1) by subtracting 1.0
            sfpi::vFloat rand_01 = sfpi::reinterpret<sfpi::vFloat>(one_to_two) - sfpi::vConst1;
            // Map to [lower, upper]
            sfpi::vFloat slope = lower + rand_01 * range;
            result = x * slope;
        }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
