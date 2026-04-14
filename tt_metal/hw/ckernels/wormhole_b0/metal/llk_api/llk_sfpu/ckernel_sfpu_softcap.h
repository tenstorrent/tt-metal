// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// Helper: compute 2^z using exp_21f algorithm (Moroz et al. 2022)
// Input z must be clamped to avoid overflow/underflow before calling.
// Returns 2^z as a vFloat.
template <bool APPROXIMATION_MODE>
inline sfpi::vFloat exp_21f(sfpi::vFloat z) {
    // Step 1: Scale by 2^23 to shift fractional bits into integer position
    z = sfpi::addexp(z, 23);

    // Step 2: Add IEEE 754 bias (0x3F800000 = 1.0f) and convert to int
    const sfpi::vFloat bias = sfpi::vFloat(0x3f800000);
    sfpi::vInt z_int = _float_to_int32_positive_(z + bias);

    // Step 3: Decompose into exponent and mantissa parts
    sfpi::vInt exp_part = sfpi::exexp(sfpi::reinterpret<sfpi::vFloat>(z_int));
    sfpi::vInt man_part = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z_int));

    // Step 4: Polynomial refinement for 2^frac(z)
    sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7f);
    sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(0xf94ee7) + man_part, 0);
    sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(0x560e) + man_part, 0);

    d2 = d1 * d2;
    sfpi::vInt frac_int = _float_to_int32_positive_(d2 * d3);

    // Step 5: Reconstruct result = mantissa_frac * 2^exponent
    sfpi::vInt result_int =
        sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(frac_int), 127U + exp_part));

    return sfpi::reinterpret<sfpi::vFloat>(result_int);
}

// tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
//         = (2^(x * log2(e)) - 2^(-x * log2(e))) / (2^(x * log2(e)) + 2^(-x * log2(e)))
//
// For small |x| (< 0.5), we use Taylor approximation: tanh(x) ≈ x - x³/3
template <bool APPROXIMATION_MODE>
inline sfpi::vFloat calculate_tanh(sfpi::vFloat x) {
    constexpr float log2e = 1.4426950408889634f;
    const sfpi::vFloat v_log2e = log2e;
    const sfpi::vFloat v_half = 0.5f;
    const sfpi::vFloat v_low_threshold = -127.0f;
    const sfpi::vFloat v_third = 0.33333333f;

    // Compute z_pos = x * log2(e) for exp(x) = 2^z_pos
    sfpi::vFloat z_pos = x * v_log2e;

    // Clamp to prevent underflow
    v_if(z_pos < v_low_threshold) { z_pos = v_low_threshold; }
    v_endif;

    sfpi::vFloat exp_pos = exp_21f<APPROXIMATION_MODE>(z_pos);

    // Compute z_neg = -x * log2(e) for exp(-x) = 2^z_neg
    sfpi::vFloat z_neg = -z_pos;

    // Clamp to prevent underflow (z_neg could be very negative for large positive x)
    v_if(z_neg < v_low_threshold) { z_neg = v_low_threshold; }
    v_endif;

    sfpi::vFloat exp_neg = exp_21f<APPROXIMATION_MODE>(z_neg);

    // tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    sfpi::vFloat numerator = exp_pos - exp_neg;
    sfpi::vFloat denominator = exp_pos + exp_neg;
    sfpi::vFloat y = numerator / denominator;

    // For small |x|, override with Taylor: tanh(x) ≈ x - x³/3
    sfpi::vFloat abs_x = sfpi::setsgn(x, 0);
    v_if(abs_x < v_half) {
        sfpi::vFloat x_sq = x * x;
        y = x - x_sq * x * v_third;
    }
    v_endif;

    return y;
}

// softcap(x, cap) = cap * tanh(x / cap)
//
// Implementation notes:
// - cap parameter is passed as FP16_B format uint32_t and converted to vFloat
// - Uses tanh helper function for the core computation
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_softcap_(const int iterations, std::uint32_t param0, std::uint32_t param1, std::uint32_t param2) {
    // param0 contains the cap value in FP16_B format
    sfpi::vFloat cap = sfpi::s2vFloat16b(param0);

#pragma GCC unroll 0
    for (int d = 0; d < iterations; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // Compute x / cap
        sfpi::vFloat x_scaled = x / cap;

        // Compute tanh(x / cap)
        sfpi::vFloat tanh_result = calculate_tanh<APPROXIMATION_MODE>(x_scaled);

        // Compute cap * tanh(x / cap)
        sfpi::vFloat y = cap * tanh_result;

        // Convert to bfloat16 for deterministic rounding
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));

        sfpi::dst_reg[0] = y;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
