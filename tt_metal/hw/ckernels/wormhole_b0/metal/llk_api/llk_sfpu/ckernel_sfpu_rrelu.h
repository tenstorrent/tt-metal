// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// RReLU(x) = x if x >= 0
//            a * x if x < 0
//
// Eval mode (lower == upper):  a = lower = (lower + upper) / 2
// Training mode (lower != upper): a sampled pseudo-randomly in [lower, upper]
//   using mantissa bits of the input as a deterministic hash.
//
// Parameters are passed as uint32_t bit-cast floats.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_rrelu(uint32_t lower_u, uint32_t upper_u) {
    union {
        uint32_t u;
        float f;
    } cvt0, cvt1;
    cvt0.u = lower_u;
    cvt1.u = upper_u;
    float lower = cvt0.f;
    float upper = cvt1.f;
    float range = upper - lower;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat result = x;

        v_if(x < 0.0f) {
            sfpi::vFloat slope;
            if (range == 0.0f) {
                // Eval mode: fixed slope
                slope = lower;
            } else {
                // Training mode: pseudo-random slope in [lower, upper]
                // Use mantissa bits of |x| to generate a value in [0, 1)
                sfpi::vInt xi = sfpi::reinterpret<sfpi::vInt>(sfpi::abs(x));
                sfpi::vInt rand_bits = (xi & sfpi::vInt(0x007FFFFF)) | sfpi::vInt(0x3F800000);
                sfpi::vFloat rand01 = sfpi::reinterpret<sfpi::vFloat>(rand_bits) - 1.0f;
                slope = lower + rand01 * range;
            }
            result = x * slope;
        }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
