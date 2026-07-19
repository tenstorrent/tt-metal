// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_conversions.h"
#include "sfpi.h"
#include "ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_polyval.h"

using namespace sfpi;

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
 *
 * @see Moroz et al. 2022 - "Simple Multiple Precision Algorithms for Exponential Functions"
 *      ( https://doi.org/10.1109/MSP.2022.3157460 )
 */
template <bool is_fp32_dest_acc_en = false>
sfpi_inline sfpi::vFloat _sfpu_binary_power_21f_(sfpi::vFloat base, sfpi::vFloat pow) {
    // The algorithm works in two steps:
    // 1) Compute log2(base)
    // 2) Compute base**pow = 2**(pow * log2(base))

    // Step 1: Compute log2(base)
    // Normalize base to calculation range
    sfpi::vFloat absbase = setsgn(base, 0);       // set base as positive
    sfpi::vFloat x = sfpi::setexp(absbase, 127);  // set exp to exp bias (put base in range of 1-2)

    // 3rd order polynomial approx - determined using rminimax over [1,2]
    sfpi::vFloat series_result = x * (x * (x * 0x2.44734p-4f - 0xd.e712ap-4f) + 0x2.4f5388p+0f) - 0x1.952992p+0f;

    // Convert exponent to float
    auto exp = sfpi::convert<sfpi::vSMag>(sfpi::exexp(base));
    sfpi::vFloat exp_f32 = sfpi::convert<sfpi::vFloat>(exp, sfpi::RoundMode::Nearest);

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

    z_f32 = addexp(z_f32, 23);  // equal to multiplying by 2**23
    const sfpi::vFloat bias = sfpi::vFloat(0x3f800000);
    sfpi::vInt z = _float_to_int32_positive_(z_f32 + bias);

    sfpi::vInt zii = sfpi::exexp(sfpi::as<sfpi::vFloat>(z));  // Note: z & 0x7f800000 in paper
    sfpi::vInt zif = sfpi::exman(sfpi::as<sfpi::vFloat>(z));  // Note: z & 0x007fffff in paper

    // Compute formula in Horner form
    sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7);
    sfpi::vFloat d2 =
        sfpi::convert<sfpi::vFloat>(sfpi::as<sfpi::vSMag>(sfpi::vInt(0xf94ee7) + zif), sfpi::RoundMode::Nearest);
    sfpi::vFloat d3 =
        sfpi::convert<sfpi::vFloat>(sfpi::as<sfpi::vSMag>(sfpi::vInt(0x560e) + zif), sfpi::RoundMode::Nearest);

    d2 = d1 * d2;
    zif = _float_to_int32_positive_(d2 * d3);

    // Restore exponent
    zii = sfpi::as<sfpi::vInt>(sfpi::setexp(sfpi::as<sfpi::vFloat>(zif), 127U + zii));

    sfpi::vFloat y = sfpi::as<sfpi::vFloat>(zii);

    // Post-processing: ensure that special values (e.g. 0**0, -1**0.5, ...) are handled correctly
    // Check valid base range
    auto pow_int = sfpi::convert<sfpi::vSMag16>(
        pow, sfpi::RoundMode::Nearest);  // int16 should be plenty, since large powers will approach 0/Inf
    auto pow_rounded = sfpi::convert<sfpi::vFloat>(pow_int, sfpi::RoundMode::Nearest);

    // Division by 0 when base is 0 and pow is negative => set to NaN
    v_if((absbase == 0.f) && pow < 0.f) {
        y = std::numeric_limits<float>::quiet_NaN();  // negative powers of 0 are NaN, e.g. pow(0, -1.5)
    }
    v_endif;

    v_if(base < 0.0f) {  // negative base
        // If pow is odd integer then result is negative
        // If power is even, then result is positive
        // To get the sign bit of result, we can shift last bit of pow_int to the 1st bit
        y = sfpi::setsgn2(y, pow_int);

        // Check for integer power, if it is not then overwrite result with NaN
        v_if(pow_rounded != pow) {  // negative base and non-integer power => set to NaN
            y = std::numeric_limits<float>::quiet_NaN();
        }
        v_endif;
    }
    v_endif;

    if constexpr (!is_fp32_dest_acc_en) {
        // LRegs work on float32 data. If DST is bfloat16 then SFPSTORE will truncate it.
        // This can reduce accuracy: for instance, 9**2 = 80.8 gets round to 80.5
        // rather than 81 (which would have been correct).
        // To avoid this issue, we explicitly convert to bfloat16 using round-to-nearest.
        y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
    }

    return y;
}

sfpi_inline sfpi::vFloat _sfpu_binary_power_f32_(sfpi::vFloat base, sfpi::vFloat pow) {
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
    // If m >= sqrt(2), divide by 2 and increment exponent
    v_if(m >= SQRT2) {
        // m = m * 0.5f;  // Divide by 2
        m = m * 0.5f;
        exp = exp + 1;
    }
    v_endif;

    // Transform to z = (m - 1) / (m + 1)
    sfpi::vFloat m_plus_1 = m + 1.0f;  // t in [1.707, 2.414] since m in [sqrt(2)/2, sqrt(2)]
    // 1/t: initial guess 1.0f - 0.2426406871192851f*t (linear interp on [1.7,2.4]), then Newton-Raphson y = y*(2 -
    // t*y).
    sfpi::vFloat recip = 1.0f - 0.2426406871192851f * m_plus_1;
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

    // log2(base) = ln(base)/ln(2) = exp + ln_m/ln(2). Keep the two contributions
    // separate: exp_f32 is the large integer part, ln_m*ln2inv the small fractional
    // part. Collapsing pow*log2(base) into one fp32 before 2**z squeezes out the
    // fractional mantissa bits for large |base| (the 20-27 ULP error). Instead carry
    // z = pow*log2(base) as an unevaluated double-float (z_hi, z_lo) and cancel the
    // large integer part against k=round(z) before the tail is ever rounded away.
    const sfpi::vFloat vConst1Ln2 = sfpi::vConstFloatPrgm0;
    constexpr float LN2 = 0.693147180559945309f;

    // Step 2: base**pow = 2**(pow*log2(base)).
    // The residual after the two-sum is the fp32 rounding of pow*exp_f32 itself:
    // exp_f32 is the large integer exponent and pow can carry a full 24-bit mantissa,
    // so the product needs ~30 bits and drops its low bits before the two-sum sees
    // them. exp_f32 is a small integer, so a Veltkamp split of pow makes both partial
    // products exact (12-bit half * <=7-bit integer fits a 24-bit significand); the low
    // half pow_lo*exp_f32 rides in z_lo so none of the integer term's bits are lost.
    constexpr float VELTKAMP_SPLIT = 4097.0f;  // 2**12 + 1
    sfpi::vFloat pc = pow * VELTKAMP_SPLIT;
    sfpi::vFloat pow_hi = pc - (pc - pow);
    sfpi::vFloat pow_lo = pow - pow_hi;

    sfpi::vFloat z_hi = pow_hi * exp_f32;
    sfpi::vFloat z_lo = pow_lo * exp_f32 + pow * (ln_m * vConst1Ln2);

    // Dekker FastTwoSum so k=round(z) sees the true integer part while the residual e
    // keeps the dropped tail. Exact under the precondition |z_hi| >= |z_lo| or z_hi == 0:
    // z_hi = pow*exponent(base) and z_lo carries the fractional log2 term (|z_lo| <
    // ~0.5*|pow| plus a <=2**-12*|pow| Veltkamp remainder). When exponent(base) == 0 then
    // z_hi == 0 exactly (the sum is already exact); otherwise |exponent(base)| >= 1 so
    // |z_hi| >= |pow| > |z_lo|.
    sfpi::vFloat s = z_hi + z_lo;
    sfpi::vFloat e = z_lo - (s - z_hi);
    // vConstFloatPrgm1 holds -127 (matches the original clamp); use it directly to avoid a copy.
    v_if(s < sfpi::vConstFloatPrgm1) {
        s = sfpi::vConstFloatPrgm1;
        e = 0.0f;
    }
    v_endif;

    sfpi::vInt k_int;
    sfpi::vFloat k = _sfpu_round_to_nearest_int32_(s, k_int);
    // Reduced argument (s - k) is exact by Sterbenz; add back the tail e.
    sfpi::vFloat frac = (s - k) + e;

    // 2**frac via the accurate exp helper (frac is small), then scale by 2**k.
    sfpi::vFloat y = _sfpu_exp_fp32_accurate_(frac * LN2);
    // setexp writes the 8-bit exponent field and wraps instead of saturating, so an
    // overflowing magnitude silently becomes a finite value. Detect overflow from the
    // biased exponent about to be written (>= 255 is the inf field) and clamp explicitly.
    // Checking out_exp (already needed by setexp) instead of keeping the float s live
    // across the exp helper avoids pushing this kernel past the SFPU register-allocator
    // budget (reload-insn ICE); out_exp >= 255 is equivalent to s >= 128.
    sfpi::vInt out_exp = sfpi::exexp(y, sfpi::ExponentMode::Biased) + k_int;
    y = sfpi::setexp(y, out_exp);
    v_if(out_exp >= 255) { y = std::numeric_limits<float>::infinity(); }
    v_endif;

    // Division by 0 when base is 0 and pow is negative => set to NaN (only for negative exponents)
    v_if(base == 0.f && pow < 0.f) {
        y = std::numeric_limits<float>::quiet_NaN();  // negative powers of 0 are NaN, e.g. pow(0, -1.5)
    }
    v_endif;

    v_if(base < 0.0f) {  // negative base
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
            y = std::numeric_limits<float>::quiet_NaN();
        }
        v_endif;
    }
    v_endif;

    return y;
}

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_binary_power_(sfpi::vFloat base, sfpi::vFloat pow);

// is_fp32_dest_acc_en == false
template <>
sfpi_inline sfpi::vFloat _sfpu_binary_power_<false>(sfpi::vFloat base, sfpi::vFloat pow) {
    return _sfpu_binary_power_21f_<false>(base, pow);
}

// is_fp32_dest_acc_en == true
template <>
sfpi_inline sfpi::vFloat _sfpu_binary_power_<true>(sfpi::vFloat base, sfpi::vFloat pow) {
    return _sfpu_binary_power_f32_(base, pow);
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool is_fp32_dest_acc_en = false>
inline void calculate_sfpu_binary_pow(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    for (int d = 0; d < ITERATIONS; d++) {
        // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
        constexpr uint dst_tile_size_sfpi = 32;
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        sfpi::vFloat result = _sfpu_binary_power_<is_fp32_dest_acc_en>(in0, in1);

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void sfpu_binary_pow_init() {
    sfpi::vConstFloatPrgm0 = 1.442695f;
    sfpi::vConstFloatPrgm1 = -127.0f;
}

}  // namespace sfpu
}  // namespace ckernel
