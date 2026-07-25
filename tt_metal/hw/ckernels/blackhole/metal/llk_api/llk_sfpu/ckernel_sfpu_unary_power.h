// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_conversions.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpu/ckernel_sfpu_polyval.h"

namespace ckernel {
namespace sfpu {

/**
 * @brief Computes base raised to the power of pow (base**pow)
 *
 * This function implements binary exponentiation using a polynomial approximation algorithm
 * based on "Simple Multiple Precision Algorithms for Exponential Functions [Tips & Tricks]"
 * by Moroz et al. 2022 (https://doi.org/10.1109/MSP.2022.3157460).
 * More specifically, it is the implementation of the `exp_21f` algorithm described in Section 5
 *
 * @param base The base value (sfpi::vFloat vector), can be any floating point number
 * @param pow The exponent/power value (sfpi::vFloat vector), can be any floating point number
 * @tparam IS_POSITIVE_EXPONENT If true, assumes exponent >= 0 (skips zero-base check for optimization)
 *
 * @return sfpi::vFloat Result of base**pow
 *
 * Special Cases:
 * - base = 0, pow < 0: Returns NaN (undefined)
 * - base < 0, pow = integer: Returns proper signed result (negative if odd power)
 * - base < 0, pow = non-integer: Returns NaN (complex result)
 * - Overflow/underflow: Clamped to appropriate limits
 *
 * @note This function assumes that the programmable constants are set to the following values:
 * - vConstFloatPrgm0 = 1.4426950408889634f;
 * - vConstFloatPrgm1 = -127.0f;
 * - vConstFloatPrgm2 = std::numeric_limits<float>::quiet_NaN();
 *
 * @see Moroz et al. 2022 - "Simple Multiple Precision Algorithms for Exponential Functions"
 *      ( https://doi.org/10.1109/MSP.2022.3157460 )
 */
template <bool IS_POSITIVE_EXPONENT>
sfpi_inline sfpi::vFloat _sfpu_unary_power_21f_(sfpi::vFloat base, sfpi::vFloat pow) {
    // The algorithm works in two steps:
    // 1) Compute log2(base)
    // 2) Compute base**pow = 2**(pow * log2(base))

    // Step 1: Compute log2(base)
    // Normalize base to calculation range
    sfpi::vFloat abs_base = sfpi::abs(base);       // set base as positive
    sfpi::vFloat x = sfpi::setexp(abs_base, 127);  // set exp to exp bias (put base in range of 1-2)

    // 3rd order polynomial approx - determined using rminimax over [1,2]
    vFloat series_result = PolynomialEvaluator::eval(x, -0x1.952992p+0f, 0x2.4f5388p+0f, -0xd.e712ap-4f, 0x2.44734p-4f);

    // Convert exponent to float
    sfpi::vInt exp = sfpi::exexp(base);
    sfpi::vFloat exp_f32 = sfpi::convert<sfpi::vFloat>(sfpi::convert<sfpi::vSMag>(exp), sfpi::RoundMode::Nearest);

    // De-normalize to original range
    const sfpi::vFloat vConst1Ln2 = sfpi::vConstFloatPrgm0;           // vConst1Ln2 = 1.4426950408889634f;
    sfpi::vFloat log2_result = exp_f32 + series_result * vConst1Ln2;  // exp correction: ln(1+x) + exp*ln(2)

    // Step 2: Compute base**pow = 2**(pow * log2(base))
    // If (base, exponent) => (0, +inf) or (base, exponent) => (N, -inf) then output should be 0
    // However, intermediary values can overflow, which leads to output increasing again instead of
    // staying at 0.
    // This overflow happens when z_f32 < -127. Therefore, we clamp z_f32 to -127.
    sfpi::vFloat z_f32 = pow * log2_result;
    const sfpi::vFloat low_threshold = sfpi::vConstFloatPrgm1;
    v_if(z_f32 < low_threshold) { z_f32 = low_threshold; }
    v_endif;

    // The paper relies on the following formula (c.f. Sections 1 and 5):
    // z = (bias + x * log2(a)) * N_m; where:
    // N_m = 2**23
    // bias = 0x3f800000

    // In this case, we transform the formula to:
    // z = (bias) * N_m + (x * log2(a)) * N_m
    // where (bias + N_m) = 0x3f800000
    // and (x * log2(a)) * N_m = addexp(z_f32, 23)

    // Notes:
    // - N_m being a power of 2 ensures equivalent results
    // - addexp(z_f32, 23) is used because it translates to a single-cycle SFPDIVP2
    //   instruction with immediate operand (i.e. no extra register used).
    //   (vs. 1 cycle SFPLOADI + 2 cycles MAD)

    z_f32 = sfpi::addexp(z_f32, 23);  // equal to multiplying by 2**23
    const sfpi::vFloat bias = sfpi::vFloat(0x3f800000);
    sfpi::vInt z = _float_to_int32_positive_(z_f32 + bias);

    sfpi::vInt zii = sfpi::exexp(sfpi::as<sfpi::vFloat>(z));  // Note: z & 0x7f800000 in paper
    sfpi::vInt zif = sfpi::exman(sfpi::as<sfpi::vFloat>(z));  // Note: z & 0x007fffff in paper

    // Compute formula in Horner form
    sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7);
    sfpi::vFloat d2 =
        sfpi::convert<sfpi::vFloat>(sfpi::as<vSMag>(sfpi::vInt(0xf94ee7) + zif), sfpi::RoundMode::Nearest);
    sfpi::vFloat d3 = sfpi::convert<sfpi::vFloat>(sfpi::as<vSMag>(sfpi::vInt(0x560e) + zif), sfpi::RoundMode::Nearest);

    d2 = d1 * d2;
    zif = _float_to_int32_positive_(d2 * d3);

    // Restore exponent
    zii = sfpi::as<sfpi::vInt>(sfpi::setexp(sfpi::as<sfpi::vFloat>(zif), 127U + zii));

    sfpi::vFloat y = sfpi::as<sfpi::vFloat>(zii);

    // Division by 0 when base is 0 and pow is negative => set to NaN (only for negative exponents)
    if constexpr (!IS_POSITIVE_EXPONENT) {
        v_if(abs_base == 0.f) {
            y = sfpi::vConstFloatPrgm2;  // negative powers of 0 are NaN, e.g. pow(0, -1.5)
        }
        v_endif;
    }

    // Negative base handling (for both positive and negative exponents)
    v_if(base < 0.0f) {
        // Post-processing: ensure that special values (e.g. 0**0, -1**0.5, ...) are handled correctly
        // Check valid base range
        auto pow_int = sfpi::convert<sfpi::vSMag16>(
            pow, sfpi::RoundMode::Nearest);  // int16 should be plenty, since large powers will approach 0/Inf
        auto pow_rounded = sfpi::convert<sfpi::vFloat>(pow_int, sfpi::RoundMode::Nearest);

        // If pow is odd integer then result is negative
        // If power is even, then result is positive
        y = sfpi::setsgn2(y, pow_int);

        // Check for integer power, if it is not then overwrite result with NaN
        v_if(pow_rounded != pow) {  // negative base and non-integer power => set to NaN
            y = sfpi::vConstFloatPrgm2;
        }
        v_endif;
    }
    v_endif;

    // LRegs work on float32 data. If DST is bfloat16 then SFPSTORE will truncate it.
    // This can reduce accuracy: for instance, 9**2 = 80.8 gets round to 80.5
    // rather than 81 (which would have been correct).
    // To avoid this issue, we explicitly convert to bfloat16 using round-to-nearest.
    y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);

    return y;
}

sfpi_inline sfpi::vFloat _sfpu_pow2_f32_accurate_(sfpi::vFloat z) {
    // Handle underflow
    z = sfpi::max(z, -0x7e.ffff8p0f);

    sfpi::vInt k_int;
    sfpi::vFloat k = _sfpu_round_to_nearest_int32_(z, k_int);

    // Compute val = z * ln(2), then r = val - k*ln(2) in extended precision.
    constexpr float LN2 = 0.693147180559945309f;
    constexpr float LN2_HI = -0.6931152343750000f;
    constexpr float LN2_LO = -3.19461832987e-05f;

    sfpi::vFloat val = z * LN2;
    sfpi::vFloat r_hi = k * LN2_HI + val;
    sfpi::vFloat r = k * LN2_LO + r_hi;

    sfpi::vFloat p = PolynomialEvaluator::eval(
        r, 1.0f, 1.0f, 0.5f, 1.0f / 6.0f, 1.0f / 24.0f, 1.0f / 120.0f, 1.0f / 720.0f, 1.0f / 5040.0f);

    sfpi::vFloat result = sfpi::setexp(p, sfpi::exexp(p, sfpi::ExponentMode::Biased) + k_int);

    // Handle overflow
    v_if(z >= 128.0f) { result = std::numeric_limits<float>::infinity(); }
    v_endif;

    return result;
}

// Computes 2**(z_hi + z_lo) where the argument is supplied as an unevaluated sum
// (double-float). For pow(x,y): z = y*log2(x) can be O(100) with a tiny but
// significant fractional tail. Rounding (z_hi + z_lo) into one fp32 before the
// reduction discards that tail (the 20-27 ULP error). A Dekker FastTwoSum keeps the
// residual, which is folded back into the reduced remainder r so no fractional bits
// are lost. FastTwoSum is exact under the precondition |z_hi| >= |z_lo| or z_hi == 0,
// which both pow callers satisfy: z_hi = pow*exponent(base) and z_lo carries the
// fractional log2 term (|z_lo| < ~0.5*|pow| plus a <=2**-12*|pow| Veltkamp remainder).
// When exponent(base) == 0 then z_hi == 0 exactly (the sum is already exact);
// otherwise |exponent(base)| >= 1 so |z_hi| >= |pow| > |z_lo|.
sfpi_inline sfpi::vFloat _sfpu_pow2_f32_accurate_hilo_(sfpi::vFloat z_hi, sfpi::vFloat z_lo) {
    sfpi::vFloat s = z_hi + z_lo;
    sfpi::vFloat e = z_lo - (s - z_hi);

    s = sfpi::max(s, -0x7e.ffff8p0f);

    sfpi::vInt k_int;
    sfpi::vFloat k = _sfpu_round_to_nearest_int32_(s, k_int);

    // ln2 split so that f*LN2_HI is exact for the small reduced |f| <= 0.5.
    constexpr float LN2_HI = 0.693359375f;
    constexpr float LN2_LO = -2.12194440e-4f;

    // Reduced fractional argument f = (s - k) + e. (s - k) is exact by Sterbenz since
    // k is the nearest integer to s and |s| < 2**23; adding the residual e restores the
    // fractional tail that a single-fp32 (z_hi+z_lo) would have discarded.
    sfpi::vFloat f = (s - k) + e;

    sfpi::vFloat r = f * LN2_HI + f * LN2_LO;

    sfpi::vFloat p = PolynomialEvaluator::eval(
        r, 1.0f, 1.0f, 0.5f, 1.0f / 6.0f, 1.0f / 24.0f, 1.0f / 120.0f, 1.0f / 720.0f, 1.0f / 5040.0f);

    sfpi::vFloat result = sfpi::setexp(p, sfpi::exexp(p, sfpi::ExponentMode::Biased) + k_int);

    v_if(s >= 128.0f) { result = std::numeric_limits<float>::infinity(); }
    v_endif;

    return result;
}

template <bool IS_POSITIVE_EXPONENT>
sfpi_inline sfpi::vFloat _sfpu_unary_power_61f_updated_(const sfpi::vFloat& base, const sfpi::vFloat& pow) {
    // The algorithm works in two steps:
    // 1) Compute log2(base)
    // 2) Compute base**pow = 2**(pow * log2(base))

    // Step 1: Compute log2(base) using improved log
    // Normalize base to calculation range
    sfpi::vFloat abs_base = sfpi::abs(base);
    sfpi::vFloat m = sfpi::setexp(abs_base, 127);
    sfpi::vInt exp = sfpi::exexp(abs_base);

    // Range reduction: ensure m in [sqrt(2)/2, sqrt(2)] ≈ [0.707, 1.414]
    constexpr float SQRT2 = 1.4142135381698608f;
    v_if(m >= SQRT2) {
        m = sfpi::addexp(m, -1);
        exp = exp + 1;
    }
    v_endif;

    // Transform to z = (m - 1) / (m + 1)
    sfpi::vFloat m_plus_1 = m + 1.0f;  // t in [1.707, 2.414] since m in [sqrt(2)/2, sqrt(2)]
    // 1/t: initial guess 1.003f - 0.244f*t (linear interp on [1.7,2.4]), then Newton-Raphson y = y*(2 - t*y).
    sfpi::vFloat recip = 1.003f - 0.244f * m_plus_1;
    recip = recip * (2.0f - m_plus_1 * recip);  // 1st NR
    recip = recip * (2.0f - m_plus_1 * recip);  // 2nd NR
    // 3rd NR: two NR iterations leave a ~2 ULP reciprocal residual that, after the
    // atanh(z) log series and pow*log2 multiply, is the floor keeping 2.5 at 4 ULP.
    // One more quadratically-convergent step drives 1/(m+1) to full fp32 precision.
    recip = recip * (2.0f - m_plus_1 * recip);  // 3rd NR for float32
    // z = (m-1)*recip written as a single fused multiply-add (m*recip - recip), one
    // instruction instead of a separate (m-1) subtract plus a multiply.
    sfpi::vFloat z = m * recip - recip;

    // Compute z**2 for polynomial evaluation
    sfpi::vFloat z2 = z * z;
    // Polynomial approximation using odd powers
    sfpi::vFloat p = PolynomialEvaluator::eval(
        z2, 1.0f, 0.3333333333333333f, 0.2f, 0.14285714285714285f, 0.1111111111111111f, 0.09090909090909091f);
    sfpi::vFloat ln_m = 2.0f * (z * p);

    sfpi::vFloat exp_f32 = sfpi::convert<sfpi::vFloat>(sfpi::convert<sfpi::vSMag>(exp), sfpi::RoundMode::Nearest);

    // z = pow*log2(base) = pow*exp_f32 (large integer-ish part) + pow*(ln_m*ln2inv)
    // (fractional part). Summing both into one fp32 before 2**z squeezes out the
    // fractional mantissa bits for large |base| -- the source of the 20-27 ULP error.
    // Carry z as an unevaluated double-float (z_hi, z_lo); the helper's Knuth two-sum
    // preserves the fractional tail across the k=round(z) reduction.
    const sfpi::vFloat vConst1Ln2 = sfpi::vConstFloatPrgm0;

    // The residual after the two-sum is the fp32 rounding of the product pow*exp_f32
    // itself. exp_f32 is the (large) integer exponent of base and pow can carry a full
    // 24-bit mantissa (e.g. 1.7984), so pow*exp_f32 needs ~30 bits and drops its low
    // bits before the two-sum can see them. Because exp_f32 is a small integer, a
    // Veltkamp split of pow makes both partial products exact (a 12-bit half times a
    // <=7-bit integer fits a 24-bit significand):
    //   pow*exp_f32 = pow_hi*exp_f32 + pow_lo*exp_f32   (both exact)
    // pow_lo*exp_f32 rides in z_lo, so no bit of the large integer term is lost.
    // Splitting only pow (not the full Dekker two-product on log2(base)) keeps the
    // simultaneously-live vector count within the SFPU register budget.
    constexpr float VELTKAMP_SPLIT = 4097.0f;  // 2**12 + 1
    sfpi::vFloat pc = pow * VELTKAMP_SPLIT;
    sfpi::vFloat pow_hi = pc - (pc - pow);
    sfpi::vFloat pow_lo = pow - pow_hi;

    sfpi::vFloat z_hi = pow_hi * exp_f32;
    sfpi::vFloat z_lo = pow_lo * exp_f32 + pow * (ln_m * vConst1Ln2);
    sfpi::vFloat y = _sfpu_pow2_f32_accurate_hilo_(z_hi, z_lo);

    // Division by 0 when base is 0 and pow is negative => set to NaN (only for negative exponents)
    if constexpr (!IS_POSITIVE_EXPONENT) {
        v_if(abs_base == 0.f) {
            y = sfpi::vConstFloatPrgm2;  // negative powers of 0 are NaN, e.g. pow(0, -1.5)
        }
        v_endif;
    }

    v_if(base < 0.0f) {  // negative base
        // Post-processing: ensure that special values (e.g. 0**0, -1**0.5, ...) are handled correctly
        // Check valid base range
        auto pow_int = sfpi::convert<sfpi::vSMag16>(
            pow, sfpi::RoundMode::Nearest);  // int16 should be plenty, since large powers will approach 0/Inf
        auto pow_rounded = sfpi::convert<sfpi::vFloat>(pow_int, sfpi::RoundMode::Nearest);

        // If pow is odd integer then result is negative
        // If power is even, then result is positive
        // To get the sign bit of result, we can shift last bit of pow_int to the 1st bit
        y = sfpi::setsgn2(y, pow_int);

        // Check for integer power, if it is not then overwrite result with NaN
        v_if(pow_rounded != pow) {  // negative base and non-integer power => set to NaN
            y = sfpi::vConstFloatPrgm2;
        }
        v_endif;
    }
    v_endif;

    return y;
}

template <int ITERATIONS>
inline void _sfpu_unary_power_bf16_(const uint32_t exponent) {
    // Convert exponent to float
    const float pow_scalar = Converter::as_float(exponent);
    const sfpi::vFloat pow = pow_scalar;

    if (pow_scalar >= 0.0f) {
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat base = sfpi::dst_reg[0];
            sfpi::dst_reg[0] = _sfpu_unary_power_21f_<true>(base, pow);
            sfpi::dst_reg++;
        }
    } else {
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat base = sfpi::dst_reg[0];
            sfpi::dst_reg[0] = _sfpu_unary_power_21f_<false>(base, pow);
            sfpi::dst_reg++;
        }
    }
}

template <int ITERATIONS>
inline void _sfpu_unary_power_fp32_(const uint32_t exponent) {
    // Convert exponent to float
    const float pow_scalar = Converter::as_float(exponent);
    const sfpi::vFloat pow = pow_scalar;

    if (pow_scalar >= 0.0f) {
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat base = sfpi::dst_reg[0];
            sfpi::dst_reg[0] = _sfpu_unary_power_61f_updated_<true>(base, pow);
            sfpi::dst_reg++;
        }
    } else {
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat base = sfpi::dst_reg[0];
            sfpi::dst_reg[0] = _sfpu_unary_power_61f_updated_<false>(base, pow);
            sfpi::dst_reg++;
        }
    }
}

inline void power_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

/**
 * @brief Compute power operation
 *
 * @param exponent The exponent as IEEE 754 float bits (reinterpreted as uint32_t)
 */
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_unary_power(const uint32_t exponent) {
    if constexpr (is_fp32_dest_acc_en) {
        _sfpu_unary_power_fp32_<ITERATIONS>(exponent);
    } else {
        _sfpu_unary_power_bf16_<ITERATIONS>(exponent);
    }
}

/**
 * @brief Compute power operation using iterative approach
 *
 * @param exponent Non-negative integer exponent value
 */
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_power_iterative(const uint32_t exponent) {
    // iterative approach for positive integer exponents
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        if (exponent == 0) {
            sfpi::dst_reg[0] = 1.0f;
        } else {
            sfpi::vFloat result = in;
            uint32_t exp = exponent - 1;

            while (exp > 0) {
                if (exp & 1) {
                    result *= in;
                }
                in *= in;
                exp >>= 1;
            }
            sfpi::dst_reg[0] = result;
        }
        sfpi::dst_reg++;
    }
}

inline void sfpu_unary_pow_init() {
    sfpi::vConstFloatPrgm0 = 1.4426950408889634f;
    sfpi::vConstFloatPrgm1 = -127.0f;
    sfpi::vConstFloatPrgm2 = std::numeric_limits<float>::quiet_NaN();
}

}  // namespace sfpu
}  // namespace ckernel
