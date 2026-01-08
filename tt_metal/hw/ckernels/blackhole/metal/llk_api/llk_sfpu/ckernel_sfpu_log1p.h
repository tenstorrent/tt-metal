// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_log.h"

namespace ckernel {
namespace sfpu {

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat calculate_log1p_fp32(sfpi::vFloat a) {
    const float LOG_TWO = 0.693147182f;       // 0x1.62e430p-1
    const float TWO_TO_M23 = 1.19209290e-7f;  // 0x1.0p-23

    sfpi::vFloat u;
    sfpi::vFloat s;
    sfpi::vFloat m;
    sfpi::vFloat r;
    sfpi::vFloat t;

    u = a + sfpi::vConst1;
    r = std::numeric_limits<float>::quiet_NaN();

    v_if(u >= 0.0f) {
        sfpi::vFloat three_quarters = 0.75f;
        sfpi::vInt e = sfpi::reinterpret<sfpi::vInt>(three_quarters);

        e = sfpi::reinterpret<sfpi::vInt>(u) - e;
        e = sfpi::reinterpret<sfpi::vInt>(sfpi::setman(sfpi::reinterpret<sfpi::vFloat>(e), 0));

        m = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vInt>(a) - e);
        sfpi::vFloat four = 4.0f;
        s = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vInt>(four) - e);  // s' in [2**-126,2**26]

        // t = 0.25f * s + sfpi::vConstNeg1;
        sfpi::vFloat quarter = 0.25f;
        sfpi::vFloat neg1 = sfpi::vConstNeg1;
        t = __builtin_rvtt_sfpmad(quarter.get(), s.get(), neg1.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        r = -0.04541015625f;  //-4.54559326e-2f;          // -0x1.746000p-5

        m = m + t;
        t = 0.10546875;  // 1.05529785e-1f;           //  0x1.b04000p-4

        // approximate log(1+m) on [-0.25, 0.5]
        s = m * m;
        r = r * s + -1.32279143e-1f;  // -0x1.0ee85ep-3
        t = t * s + 1.44911006e-1f;   //  0x1.28c71ap-3
        r = r * s + -1.66416913e-1f;  // -0x1.54d264p-3
        t = t * s + 1.99886635e-1f;   //  0x1.995e2ap-3
        r = r * s + -2.50001878e-1f;  // -0x1.00007ep-2
        sfpi::vFloat infinity = std::numeric_limits<float>::infinity();
        r = t * m + r;
        r = r * m + 3.33335280e-1f;   //  0x1.5555d8p-2
        r = r * m + -5.00000000e-1f;  // -0x1.000000p-1
        sfpi::vFloat e_float = sfpi::int32_to_float(__builtin_rvtt_sfpcast(e.get(), sfpi::SFPCAST_MOD1_INT32_TO_SM32));
        // sfpi::vFloat e_float = sfpi::setsgn(sfpi::int32_to_float(sfpi::abs(e)), sfpi::reinterpret<sfpi::vFloat>(e));
        r = r * s + m;
        r = e_float * (LOG_TWO * TWO_TO_M23) + r;

        // since u>=0, safely checks for u == NaN or u == inf
        v_if(sfpi::reinterpret<sfpi::vInt>(u) >= sfpi::reinterpret<sfpi::vInt>(infinity)) { r = u; }
        v_endif;
    }
    v_endif;

    return r;
}

/*
 * This function implements ln(1+x) using Chebyshev approximation for bfloat16.
 * Uses 3rd order Chebyshev polynomial approximation with range reduction.
 *
 * @tparam FAST_APPROX If true, skip NaN check for negative inputs
 * @tparam is_fp32_dest_acc_en If false, round result to bfloat16
 * @param val The input value x
 * @return Result of ln(1+val)
 */
template <bool FAST_APPROX, bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat calculate_log1p_bf16(sfpi::vFloat val) {
    sfpi::vFloat abs_x = sfpi::abs(val);
    sfpi::vFloat result;
    v_if(abs_x < 0.0078125) {  // use 2^(-7) as threshold value
        result = val;          // log(1+x) ~ x for x < 0.01
    }
    v_else {
        sfpi::vFloat in = val + sfpi::vConst1;
        result = calculate_log_body<FAST_APPROX, /*HAS_BASE_SCALING*/ false, /*is_fp32_dest_acc_en*/ true>(in, 0);
    }
    v_endif;

    if constexpr (!is_fp32_dest_acc_en) {
        result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
    }

    return result;
}

/*
 * This function implements ln(1+x) with accurate computation for small x.
 * Algorithm based on log1p_fp32 from operations.py:
 * 1. Handle special cases (x < -1, x == -1, infinity, NaN)
 * 2. For |x| < 0.3: Use 9-term polynomial series to avoid catastrophic cancellation
 * 3. For |x| >= 0.3: Use standard ln(1+x) computation (reuses blackhole's calculate_log_f32_body logic)
 *
 * The threshold of 0.3 and 9 terms work well enough to provide good accuracy
 * across the entire domain.
 *
 * @param val The input value (sfpi::vFloat vector), can be any floating point number > -1
 * @return sfpi::vFloat Result of ln(1+val)
 */
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat calculate_log1p_fp32_old(sfpi::vFloat val) {
    sfpi::vFloat result = sfpi::vConst0;

    // Check for special cases
    sfpi::vInt exp = sfpi::exexp(val);  // Get debiased exponent

    v_if(sfpi::reinterpret<sfpi::vInt>(val) == 0x7F800000) {
        // If input is infinity, return infinity
        result = std::numeric_limits<float>::infinity();
    }
    v_elseif(exp == 128 || val < -1.f) {                   // NaN or negative input -> NaN
        result = std::numeric_limits<float>::quiet_NaN();  // returns nan
    }
    v_elseif(val == -1.f) {
        // x = -1 input -> -inf
        result = -std::numeric_limits<float>::infinity();
    }
    v_else {
        // Normal computation
        sfpi::vFloat abs_val = sfpi::abs(val);

        constexpr float THRESHOLD = 0.3f;
        v_if(abs_val < THRESHOLD) {
            // For |x| < 0.3, use 9-term polynomial series (with 10 coefficients  including constant term)
            // to avoid catastrophic cancellation
            // Coefficients were computed using Sollya with the following command:
            // > fpminimax(log(x+1), [|1,2,3,4,5,6,7,8,9|], [|single...|], [-0.3; -2^(-20)] + [2^(-20); 0.3], relative);
            result = PolynomialEvaluator::eval(
                val,
                sfpi::vConst0,
                sfpi::vConst1,
                -0.4999997317790985107421875f,
                0.333332836627960205078125f,
                -0.250040113925933837890625f,
                0.20005328953266143798828125f,
                -0.1650786101818084716796875f,
                0.14109231531620025634765625f,
                -0.14774705469608306884765625f,
                0.133655369281768798828125);
        }
        v_else {
            // The following is the same approximation as calculate_log_f32_body from ckernel_sfpu_log.h.
            // Ideally, we would like to call calculate_log_f32_body() directly.
            // However, doing so leads to 'register spilling errors' due to interaction with the
            // polynomial evaluation near 0.

            // For |x| >= 0.3, use standard ln(1+x) computation
            sfpi::vFloat one_plus_x = sfpi::vConst1 + val;

            // Extract exponent (debiased) from (1+x)
            exp = sfpi::exexp(one_plus_x);

            // Extract mantissa and construct m in [1, 2)
            // Use setexp to normalize to [1, 2) range by setting exponent to 127 (bias)
            sfpi::vFloat m = sfpi::setexp(one_plus_x, 127);

            // Step 2: Range reduction
            // If m >= sqrt(2), divide by 2 and increment exponent
            // This ensures m is in [sqrt(2)/2, sqrt(2)] ≈ [0.707, 1.414]
            // Use vConstFloatPrgm1 which is set to sqrt(2) in log_init
            v_if(m >= sfpi::vConstFloatPrgm1) {
                m = m * 0.5f;   // Divide by 2
                exp = exp + 1;  // Increment exponent
            }
            v_endif;

            // Step 3: Transform to z = (m - 1) / (m + 1)
            // This maps m ∈ [0.707, 1.414] to z ∈ [-0.172, 0.172]
            // ln(m) = 2 × (z + z³/3 + z⁵/5 + z⁷/7 + ...)
            sfpi::vFloat m_minus_1 = m - sfpi::vConst1;
            sfpi::vFloat m_plus_1 = m + sfpi::vConst1;

            // Compute z = (m - 1) / (m + 1) using reciprocal
            // z = m_minus_1 * (1 / m_plus_1)
            sfpi::vFloat m_plus_1_recip = _sfpu_reciprocal_<2>(m_plus_1);
            sfpi::vFloat z = m_minus_1 * m_plus_1_recip;

            // Compute z**2 for polynomial evaluation
            sfpi::vFloat z2 = z * z;

            // Step 4: Polynomial approximation using odd powers
            // ln(m) = 2z(1 + (z**2)/3 + (z**4)/5 + (z**6)/7 + (z**8)/9 + (z**10)/11)
            // Using Horner's method on z² with coefficients
            sfpi::vFloat p = PolynomialEvaluator::eval(
                z2,
                sfpi::vConst1,
                0.3333333333333333f,
                0.2f,
                0.14285714285714285f,
                0.1111111111111111f,
                .09090909090909091f);

            // Final computation: ln(m) = 2 * z * p
            sfpi::vFloat ln_m = 2.0f * (z * p);

            // Convert exponent to float using sign-magnitude format
            sfpi::vInt signmag_exp = exp;

            // We want to convert exponent to floating point using int32 -> float conversion.
            // However, int32_to_float takes a sign-magnitude
            // This is not an issue for positive numbers (same representation)
            // For negative numbers, we need to explicitly convert to sign-magnitude format
            v_if(exp < 0) {
                // Compute absolute value: ~exp + 1 (two's complement negation)
                sfpi::vInt exp_abs = ~exp + 1;
                // Convert to sign-magnitude negative: setsgn(value, 1) sets MSB to 1
                signmag_exp = sfpi::setsgn(exp_abs, 1);
            }
            v_endif;

            // Convert to float - int32_to_float handles sign-magnitude format correctly
            sfpi::vFloat expf = sfpi::int32_to_float(signmag_exp, 0);

            // Step 5: Combine: ln(1+x) = exp×ln(2) + ln(m)
            // Use vConstFloatPrgm2 which is set to ln(2) in log_init
            result = expf * sfpi::vConstFloatPrgm2 + ln_m;
        }
        v_endif;
    }
    v_endif;

    if constexpr (!is_fp32_dest_acc_en) {
        // Convert to bfloat16 if needed using round-to-nearest-even
        result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
    }

    return result;
}

/**
 * @tparam APPROXIMATION_MODE If true, use approximation mode (for consistency with log kernel)
 * @tparam FAST_APPROX If true, skip NaN check for negative inputs
 * @tparam is_fp32_dest_acc_en If true, DEST registers are fp32, and output does not need to be rounded to bfloat16
 * @tparam ITERATIONS Number of iterations for given face
 */
template <bool APPROXIMATION_MODE, bool FAST_APPROX, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_log1p() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat result;
        if constexpr (is_fp32_dest_acc_en) {
            result = calculate_log1p_fp32<is_fp32_dest_acc_en>(in);
        } else {
            result = calculate_log1p_bf16<FAST_APPROX, is_fp32_dest_acc_en>(in);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

/**
 * @tparam APPROXIMATION_MODE If true, use approximation mode (for consistency with log kernel)
 * @tparam FAST_APPROX If true, skip NaN check for negative inputs
 * @tparam is_fp32_dest_acc_en If true, DEST registers are fp32, and output does not need to be rounded to bfloat16
 */
template <bool APPROXIMATION_MODE, bool FAST_APPROX, bool is_fp32_dest_acc_en>
inline void log1p_init() {
    if constexpr (!is_fp32_dest_acc_en) {
        log_init<APPROXIMATION_MODE, FAST_APPROX, is_fp32_dest_acc_en>();
    } else {
        // _init_sfpu_reciprocal_ sets vConstFloatPrgm0 to 2.0f
        _init_sfpu_reciprocal_</*approximation_mode*/ false>();
        // But we can use 2 other programmable constants:
        sfpi::vConstFloatPrgm1 = 1.4142135381698608f;   // sqrt(2)
        sfpi::vConstFloatPrgm2 = 0.69314718246459961f;  // log(2)
    }
}

}  // namespace sfpu
}  // namespace ckernel
