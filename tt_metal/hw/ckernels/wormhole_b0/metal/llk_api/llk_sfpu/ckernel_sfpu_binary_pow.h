// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_conversions.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_polyval.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

/**
 * @brief Computes base raised to the power of pow (base**pow), for a bfloat16 dest
 *
 * Cheaper than the fp32 path: with only 8 mantissa bits to fill, a single fp32
 * z = pow*log2(base) is precise enough, so the double-float argument split that
 * _sfpu_binary_power_f32_ needs can be dropped. log2 does still have to be
 * accurate relative to its own magnitude, which is what the range reduction and
 * the factored series are for.
 *
 * @param base The base value (sfpi::vFloat vector), can be any floating point number
 * @param pow The exponent/power value (sfpi::vFloat vector), can be any floating point number
 *
 * @return sfpi::vFloat Result of base**pow, rounded to bfloat16
 *
 * Special Cases:
 * - base = 0, pow < 0: Returns NaN (undefined)
 * - base < 0, pow = integer: Returns proper signed result (negative if odd power)
 * - base < 0, pow = non-integer: Returns NaN (complex result)
 * - Overflow saturates to +/-inf, magnitudes below 2**-126 flush to zero
 *
 * @note This function assumes that the programmable constants are set to the following values:
 * - vConstFloatPrgm0 = 1.4426950408889634f;
 */
sfpi_inline sfpi::vFloat _sfpu_binary_power_bf16_(sfpi::vFloat base, sfpi::vFloat pow) {
    // Step 1: Compute log2(base)
    sfpi::vFloat abs_base = sfpi::abs(base);
    sfpi::vFloat m = sfpi::setexp(abs_base, 127);
    sfpi::vInt exp = sfpi::exexp(abs_base);

    // Reduce to m in [sqrt(2)/2, sqrt(2)). Otherwise a base just under 1 comes out
    // as exponent -1 with log2(m) near 1, and the two cancel.
    constexpr float SQRT2 = 1.4142135381698608f;
    v_if(m >= SQRT2) {
        m = sfpi::addexp(m, -1);
        exp = exp + 1;
    }
    v_endif;

    // ln(1+u)/u over u in [sqrt(2)/2-1, sqrt(2)-1]. Fitting ln(m)/u rather than
    // ln(m) keeps the error proportional to u, so it dies off as base -> 1.
    sfpi::vFloat u = m - 1.0f;
    sfpi::vFloat p = PolynomialEvaluator::eval(
        u, 1.00001132f, -0.499859631f, 0.332090169f, -0.254516661f, 0.225501433f, -0.146625668f);

    const sfpi::vFloat vConst1Ln2 = sfpi::vConstFloatPrgm0;  // vConst1Ln2 = 1.4426950408889634f;
    sfpi::vFloat exp_f32 = sfpi::convert<sfpi::vFloat>(sfpi::convert<sfpi::vSMag>(exp), sfpi::RoundMode::Nearest);
    sfpi::vFloat z = pow * (exp_f32 + (u * p) * vConst1Ln2);

    // Step 2: Compute base**pow = 2**z
    // Anything that neither overflows nor underflows has |z| <= 128, so the clamp
    // is free and keeps the rounding below sane for infinities and absurd powers.
    z = sfpi::min(sfpi::max(z, -128.0f), 128.0f);

    sfpi::vInt k_int;
    sfpi::vFloat k = _sfpu_round_to_nearest_int32_(z, k_int);

    // 2**f for the reduced |f| <= 0.5
    sfpi::vFloat q =
        PolynomialEvaluator::eval(z - k, 1.00000012f, 0.69310534f, 0.240218654f, 0.0560058281f, 0.00968570821f);

    // Scale by 2**k. setexp wraps the 8-bit exponent field instead of saturating,
    // so both ends need an explicit check.
    sfpi::vInt out_exp = sfpi::exexp(q, sfpi::ExponentMode::Biased) + k_int;
    sfpi::vFloat y = sfpi::setexp(q, out_exp);
    v_if(out_exp >= 255) { y = std::numeric_limits<float>::infinity(); }
    v_elseif(out_exp <= 0) { y = 0.0f; }
    v_endif;

    // Division by 0 when base is 0 and pow is negative => set to NaN
    v_if(abs_base == 0.f && pow < 0.f) {
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
        // To get the sign bit of result, we can shift last bit of pow_int to the 1st bit
        y = sfpi::setsgn2(y, pow_int);

        // Check for integer power, if it is not then overwrite result with NaN
        v_if(pow_rounded != pow) {  // negative base and non-integer power => set to NaN
            y = std::numeric_limits<float>::quiet_NaN();
        }
        v_endif;
    }
    v_endif;

    // LRegs work on float32 data. If DST is bfloat16 then SFPSTORE will truncate it.
    // This can reduce accuracy: for instance, 9**2 = 80.8 gets round to 80.5
    // rather than 81 (which would have been correct).
    // To avoid this issue, we explicitly convert to bfloat16 using round-to-nearest.
    return sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
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
    return _sfpu_binary_power_bf16_(base, pow);
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
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstFloatPrgm0 = 1.4426950408889634f;
    sfpi::vConstFloatPrgm1 = -127.0f;
}

}  // namespace sfpu
}  // namespace ckernel
