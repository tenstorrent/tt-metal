// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

// Include sinh kernel for exp_21f helper
#include "ckernel_sfpu_sinh.h"

namespace ckernel::sfpu {

// softcap(x, cap) = cap * tanh(x / cap)
//
// Algorithm:
//   z = x * inv_cap   (precomputed on host)
//   tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
//
// For small |z| < 0.5: use Taylor series tanh(z) ≈ z - z³/3
//   to avoid catastrophic cancellation in the exp subtraction.
//
// Division is computed via Newton-Raphson reciprocal:
//   1. Decompose denominator into mantissa m ∈ [1,2) and exponent e
//   2. Linear initial estimate: 1/m ≈ 1.4571 - 0.5*m (~3.5 bits)
//   3. Two NR iterations on mantissa: y' = y*(2 - m*y) (~14 bits)
//   4. Apply exponent: result = refined_inv_m * 2^(-e)
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void _calculate_softcap_(const int iterations, std::uint32_t param0, std::uint32_t param1) {
    // param0 = BF16 bits of cap
    // param1 = BF16 bits of 1/cap
    sfpi::vFloat v_cap = sfpi::s2vFloat16b(param0);
    sfpi::vFloat v_inv_cap = sfpi::s2vFloat16b(param1);

    constexpr float log2e = 1.4426950408889634f;
    const sfpi::vFloat v_log2e = log2e;
    const sfpi::vFloat v_low_threshold = -127.0f;
    const sfpi::vFloat v_half = 0.5f;
    const sfpi::vFloat v_third = 0.33333334f;

#pragma GCC unroll 0
    for (int d = 0; d < iterations; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // z = x / cap
        sfpi::vFloat z = x * v_inv_cap;

        // Compute exp(z) and exp(-z) using exp_21f
        sfpi::vFloat z_log2 = z * v_log2e;
        v_if(z_log2 < v_low_threshold) { z_log2 = v_low_threshold; }
        v_endif;
        sfpi::vFloat exp_pos = exp_21f<APPROXIMATION_MODE>(z_log2);

        sfpi::vFloat neg_z_log2 = -z_log2;
        v_if(neg_z_log2 < v_low_threshold) { neg_z_log2 = v_low_threshold; }
        v_endif;
        sfpi::vFloat exp_neg = exp_21f<APPROXIMATION_MODE>(neg_z_log2);

        // tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
        sfpi::vFloat numer = exp_pos - exp_neg;
        sfpi::vFloat denom = exp_pos + exp_neg;

        // Newton-Raphson reciprocal of denom
        // Step 1: Decompose denom = m * 2^e
        sfpi::vInt e_int = sfpi::exexp(denom);
        sfpi::vFloat m = sfpi::setexp(denom, 127);  // mantissa in [1, 2)

        // Step 2: Linear initial estimate of 1/m (~3.5 bits accuracy)
        sfpi::vFloat inv_m = 1.4571f - m * 0.5f;

        // Step 3: Two NR iterations: y' = y * (2 - m * y)
        inv_m = inv_m * (2.0f - m * inv_m);  // ~7 bits
        inv_m = inv_m * (2.0f - m * inv_m);  // ~14 bits

        // Step 4: Apply exponent: 1/denom = inv_m * 2^(-e)
        sfpi::vFloat pow2_neg_e = sfpi::setexp(sfpi::vConst1, 127U + (-e_int));
        sfpi::vFloat rcp = inv_m * pow2_neg_e;

        sfpi::vFloat tanh_z = numer * rcp;

        // Override for small |z|: Taylor series tanh(z) ≈ z - z³/3
        // Avoids catastrophic cancellation when exp(z) ≈ exp(-z) ≈ 1
        sfpi::vFloat abs_z = sfpi::setsgn(z, 0);
        v_if(abs_z < v_half) {
            sfpi::vFloat z_sq = z * z;
            tanh_z = z - z_sq * z * v_third;
        }
        v_endif;

        // Final: softcap(x) = cap * tanh(x/cap)
        sfpi::vFloat result = v_cap * tanh_z;

        // Round to bf16 for deterministic output
        result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void softcap_init() {
    // No programmable constants needed — cap and inv_cap are passed as function arguments
}

}  // namespace ckernel::sfpu
