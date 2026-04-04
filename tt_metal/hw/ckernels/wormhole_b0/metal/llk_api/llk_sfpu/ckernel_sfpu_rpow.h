// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

#include "ckernel_sfpu_converter.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// rpow(x) = base^x  where base is a scalar float parameter
//
// Algorithm: base^x = 2^(x * log2(base))
//
// Uses the exp_21f algorithm from Moroz et al. 2022
// "Simple Multiple Precision Algorithms for Exponential Functions"
// (https://doi.org/10.1109/MSP.2022.3157460)
//
// Since base is a constant scalar, we precompute log2(base) once
// and then compute 2^(x * log2_base) for each element.

namespace {
inline uint32_t float_to_bits(float f) {
    union {
        float fval;
        uint32_t uval;
    } conv;
    conv.fval = f;
    return conv.uval;
}
}  // namespace

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_rpow(const uint32_t base_val) {
    // Decode base parameter from IEEE 754 bits
    const float base_scalar = Converter::as_float(base_val);
    const float abs_base = base_scalar < 0.0f ? -base_scalar : base_scalar;

    // Precompute log2(|base|) as a scalar float
    // IEEE 754: float = 2^exp * mantissa, so log2(float) = exp + log2(mantissa)
    // For mantissa in [1,2), use polynomial approximation

    uint32_t base_bits = float_to_bits(abs_base);
    int32_t base_exp = static_cast<int32_t>(((base_bits >> 23) & 0xFF)) - 127;
    // Normalize mantissa to [1,2) by setting exponent to 127
    uint32_t mantissa_bits = (base_bits & 0x007FFFFF) | 0x3F800000;
    float mantissa_norm = Converter::as_float(mantissa_bits);

    // 3rd order polynomial approximation for log2(x) over [1,2)
    // Same coefficients as in _sfpu_unary_power_21f_ from ckernel_sfpu_unary_power.h
    const float c3 = 0x2.44734p-4f;
    const float c2 = -0xd.e712ap-4f;
    const float c1 = 0x2.4f5388p+0f;
    const float c0 = -0x1.952992p+0f;
    const float inv_ln2 = 1.4426950408889634f;

    float series = c0 + mantissa_norm * (c1 + mantissa_norm * (c2 + mantissa_norm * c3));
    float log2_base = static_cast<float>(base_exp) + series * inv_ln2;

    // Load precomputed log2(base) into a vector register
    const sfpi::vFloat v_log2_base = log2_base;
    const sfpi::vFloat v_low_threshold = -127.0f;

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // z = x * log2(base)
        sfpi::vFloat z_f32 = x * v_log2_base;

        // Clamp to prevent overflow: if z < -127, set to -127
        v_if(z_f32 < v_low_threshold) { z_f32 = v_low_threshold; }
        v_endif;

        // Compute 2^z using exp_21f algorithm (Moroz et al. 2022, Section 5)
        // Formula: result = reinterpret_as_float((bias + z * 2^23))
        // where bias = 0x3F800000
        z_f32 = sfpi::addexp(z_f32, 23);  // multiply by 2^23
        const sfpi::vFloat bias = sfpi::vFloat(0x3f800000);
        sfpi::vInt z = _float_to_int32_positive_(z_f32 + bias);

        sfpi::vInt zii = sfpi::exexp(sfpi::reinterpret<sfpi::vFloat>(z));   // exponent part
        sfpi::vInt zif = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));  // mantissa part

        // Compute 2^frac(z) using Horner form polynomial
        sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7);
        sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(0xf94ee7) + zif, 0);
        sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(0x560e) + zif, 0);

        d2 = d1 * d2;
        zif = _float_to_int32_positive_(d2 * d3);

        // Restore exponent: result = mantissa * 2^exponent
        zii = sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(zif), 127U + zii));

        sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(zii);

        // Handle special cases
        if (abs_base == 0.0f) {
            // base == 0: 0^x = 0 for x > 0, 1 for x == 0, inf for x < 0
            v_if(x > 0.0f) { y = 0.0f; }
            v_endif;
            v_if(x == 0.0f) { y = sfpi::vConst1; }
            v_endif;
            v_if(x < 0.0f) {
                // Use a large value to represent infinity
                y = sfpi::vFloat(std::numeric_limits<float>::infinity());
            }
            v_endif;
        } else if (base_scalar < 0.0f) {
            // Negative base: result is real only for integer exponents
            // For integer x: result = |base|^x * sign
            sfpi::vInt x_int = sfpi::float_to_int16(x, 0);
            sfpi::vFloat x_rounded = sfpi::int32_to_float(x_int, 0);

            // If x is odd integer, negate the result
            y = sfpi::setsgn(y, x_int << 31);

            // If x is not an integer, set result to NaN
            v_if(x_rounded != x) { y = sfpi::vFloat(std::numeric_limits<float>::quiet_NaN()); }
            v_endif;
        }

        // Convert to bfloat16 with round-to-nearest-even for accuracy
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));

        sfpi::dst_reg[0] = y;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void rpow_init() {
    // No programmable constants needed - log2(base) is computed from the parameter
}

}  // namespace sfpu
}  // namespace ckernel
