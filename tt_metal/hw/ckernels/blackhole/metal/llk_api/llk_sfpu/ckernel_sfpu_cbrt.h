// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ckernel {
namespace sfpu {

// cbrt(x) = x^(1/3) (cube root)
// For negative inputs: cbrt(-x) = -cbrt(x)
//
// Algorithm:
// 1. Save sign, work with absolute value
// 2. IEEE 754 magic constant for initial reciprocal-cbrt estimate:
//    y0 = reinterpret(0x548c2b4b - reinterpret(|x|) / 3)
// 3. Newton-Raphson refinement for reciprocal cube root:
//    y = y * (4/3) - (x * y^4) * (1/3)
//    (2 iterations for good precision)
// 4. Final multiply: cbrt(x) = x * y^2  where y ~ 1/cbrt(x)
// 5. Restore original sign
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_cbrt() {
    constexpr float one_third = 0.333333343f;
    constexpr float four_thirds = 1.33333337f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // Save sign and work with absolute value
        sfpi::vUInt x_bits = sfpi::reinterpret<sfpi::vUInt>(x);
        sfpi::vUInt sign_bit = x_bits & 0x80000000;
        sfpi::vFloat abs_x = sfpi::reinterpret<sfpi::vFloat>(x_bits & 0x7FFFFFFF);

        // IEEE 754 magic constant for initial reciprocal-cbrt estimate
        // y0 ~ 1/cbrt(|x|) via bit manipulation
        sfpi::vUInt abs_bits = sfpi::reinterpret<sfpi::vUInt>(abs_x);
        sfpi::vUInt est_bits = sfpi::vUInt(0x548c2b4b) - abs_bits / 3;
        sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(est_bits);

        // Newton-Raphson iteration 1 for reciprocal cube root:
        // y = y * (4/3) - (abs_x * y^4) * (1/3)
        sfpi::vFloat y2 = y * y;
        sfpi::vFloat y4 = y2 * y2;
        y = y * four_thirds - abs_x * y4 * one_third;

        // Newton-Raphson iteration 2
        y2 = y * y;
        y4 = y2 * y2;
        y = y * four_thirds - abs_x * y4 * one_third;

        // Convert from reciprocal cbrt to cbrt: cbrt(|x|) = |x| * y^2
        y2 = y * y;
        sfpi::vFloat result = abs_x * y2;

        // Handle x == 0 (avoid NaN from 0 * inf)
        v_if(abs_x == 0.0f) { result = 0.0f; }
        v_endif;

        // Restore sign: cbrt(-x) = -cbrt(x)
        sfpi::vUInt result_bits = sfpi::reinterpret<sfpi::vUInt>(result);
        result = sfpi::reinterpret<sfpi::vFloat>(result_bits | sign_bit);

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
