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
#include "ckernel_sfpu_polyval.h"

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
 * - base = 0, pow > 0: Returns 0
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
template <bool is_fp32_dest_acc_en>
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

    sfpi::vInt zii = exexp(sfpi::as<sfpi::vFloat>(z));        // Note: z & 0x7f800000 in paper
    sfpi::vInt zif = sfpi::exman(sfpi::as<sfpi::vFloat>(z));  // Note: z & 0x007fffff in paper

    // Compute formula in Horner form
    sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7);
    sfpi::vFloat d2 =
        sfpi::convert<sfpi::vFloat>(sfpi::as<sfpi::vSMag>(sfpi::vInt(0xf94ee7) + zif), sfpi::RoundMode::Nearest);
    sfpi::vFloat d3 =
        sfpi::convert<sfpi::vFloat>(sfpi::as<sfpi::vSMag>(sfpi::vInt(0x560e) + zif), sfpi::RoundMode::Nearest);

    d2 = d1 * d2;
    sfpi::vFloat d23 = d2 * d3;
    zif = sfpi::exman(d23, sfpi::MantissaMode::ImplicitOne);
    zif = sfpi::shft(zif, sfpi::exexp(d23) - 23, sfpi::ShiftMode::Logical);

    // Restore exponent
    zii = sfpi::as<sfpi::vInt>(sfpi::setexp(sfpi::as<sfpi::vFloat>(zif), 127U + zii));

    sfpi::vFloat y = sfpi::as<sfpi::vFloat>(zii);

    // Post-processing: ensure that special values (e.g. 0**0, -1**0.5, ...) are handled correctly
    // Check valid base range
    sfpi::vFloat abs_pow = sfpi::abs(pow);
    sfpi::vFloat pow_x = sfpi::min(sfpi::max(sfpi::vFloat(0.5f), abs_pow), 16777216.0f);
    sfpi::vInt pow_bits = sfpi::exman(pow_x, sfpi::MantissaMode::ImplicitOne);
    pow_bits = sfpi::shft(pow_bits, sfpi::exexp(pow_x) + 7, sfpi::ShiftMode::Logical);
    pow_bits = pow_bits + pow_bits;

    y = sfpi::copysgn(y, sfpi::as<sfpi::vFloat>(pow_bits & sfpi::as<sfpi::vInt>(base)));

    v_if(base < 0.0f) {
        // Check for integer power, if it is not then overwrite result with NaN
        v_if((pow_bits + pow_bits) != 0 && abs_pow != 0.0f) {
            y = std::numeric_limits<float>::quiet_NaN();
        }
        v_endif;
    }
    v_endif;

    // setexp/exexp map 0 to log2 = -127, so 0**p evaluates as 2**(-127p).
    // Must follow the negative-base branch: SFPU `<` is sign-bit based, so
    // v_if(base < 0) classes -0 as negative and would overwrite the 0 with the
    // complex-result NaN; -0 is a zero, not a negative base.
    // Fill 0 for every non-zero exponent, then narrow to the negative ones.
    // v_and tightens the enclosing predicate in place, so it costs one compare
    // where a second flat v_if would also save and restore the lane mask.
    // pow == 0 fails the != 0 gate and keeps 1.
    v_if(absbase == 0.f) {
        v_if(pow != 0.f) {
            y = 0.0f;
            v_and(pow < 0.f);
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

/**
 * @brief base**pow on an fp32 DST.
 *
 * Same two-step log2/exp2 shape as _sfpu_binary_power_21f_, but with a wider log. The
 * zero-base block below is fp32-only: _sfpu_binary_power_21f_ still carries the original
 * one verbatim, so on a bf16 DST pow(0, NaN) returns +0 and pow(0, pow < 0) writes NaN.
 * See #53922, deferred there on tt-llk#675.
 *
 * @param base The base value (sfpi::vFloat vector), can be any floating point number
 * @param pow The exponent/power value (sfpi::vFloat vector), can be any floating point number
 *
 * @return sfpi::vFloat Result of base**pow
 *
 * @note Two consumers, both via _sfpu_binary_power_: calculate_sfpu_binary_pow, and
 * calculate_rpow, which passes its scalar as `base`. So ttnn.rpow(y, 0.0) is pow(0, y)
 * and the zero-base semantics below apply to it too.
 *
 * Special Cases:
 * - base = +/-0, pow > 0: Returns +/-0 for finite pow, preserving the base sign only for odd integer pow; +inf returns
 * +0
 * - base = +/-0, pow < 0: Returns +/-inf for finite pow, preserving the base sign only for odd integer pow; -inf
 * returns +inf
 * - base = +/-0, pow = +/-0: Returns 1
 * - pow = +/-NaN: Returns NaN for every base
 * - base < 0, pow = integer: Returns proper signed result (negative if odd power)
 * - base < 0, pow = non-integer: Returns NaN (complex result)
 * - Overflow/underflow: Clamped to appropriate limits
 */
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
    sfpi::vFloat m_minus_1 = m - 1.0f;
    sfpi::vFloat recip = sfpi::approx_recip(m_plus_1);
    sfpi::vFloat t = 1.0f - m_plus_1 * recip;
    auto exp_sm = sfpi::convert<sfpi::vSMag>(exp);
    recip = recip * t + recip;
    t = 1.0f - m_plus_1 * recip;
    sfpi::vFloat exp_f32 = sfpi::convert<sfpi::vFloat>(exp_sm, sfpi::RoundMode::Nearest);
    recip = recip * t + recip;
    t = 1.0f - m_plus_1 * recip;
    recip = recip * t + recip;
    // z = (m-1)*recip. The subtract is kept separate rather than folded into a multiply-add:
    // the range reduction above leaves m in [sqrt(2)/2, sqrt(2)], which is inside [0.5, 2], so
    // Sterbenz's lemma makes m - 1 exact for every representable m and the only rounding left is
    // the multiply. Written as m*recip - recip it is instead a subtraction of two nearly equal
    // quantities whenever the base is near 1.0, where m*recip and recip agree to ~24 bits. SFPMAD
    // is only partially fused -- the product keeps four bits beyond fp32, not the exact product a
    // true FMA would give -- so those four bits are all that survives the cancellation. The error
    // is then scaled by pow in pow*log2(base): 250 ULP at pow = 1000, 4.4% relative at 3e6.
    sfpi::vFloat z = m_minus_1 * recip;

    // Compute z**2 for polynomial evaluation
    sfpi::vFloat z2 = z * z;
    // Polynomial approximation using odd powers
    sfpi::vFloat p = PolynomialEvaluator::eval(
        z2, 1.0f, 0.3333333333333333f, 0.2f, 0.14285714285714285f, 0.1111111111111111f);
    sfpi::vFloat ln_m = 2.0f * (z * p);


    // log2(base) = ln(base)/ln(2) = exp + ln_m/ln(2). Keep the two contributions
    // separate: exp_f32 is the large integer part, ln_m*ln2inv the small fractional
    // part. Collapsing pow*log2(base) into one fp32 before 2**z squeezes out the
    // fractional mantissa bits for large |base| (the 20-27 ULP error). Instead carry
    // z = pow*log2(base) as an unevaluated double-float (z_hi, z_lo) and cancel the
    // large integer part against k=round(z) before the tail is ever rounded away.
    const sfpi::vFloat vConst1Ln2 = sfpi::vConstFloatPrgm0;

    // Step 2: base**pow = 2**(pow*log2(base)).
    // The residual after the two-sum is the fp32 rounding of pow*exp_f32 itself:
    // exp_f32 is the large integer exponent and pow can carry a full 24-bit mantissa,
    // so the product needs ~30 bits and drops its low bits before the two-sum sees
    // them. exp_f32 is a small integer, so a Veltkamp split of pow makes both partial
    // products exact (12-bit half * <=7-bit integer fits a 24-bit significand); the low
    // half pow_lo*exp_f32 rides in z_lo so none of the integer term's bits are lost.
    sfpi::vFloat pc = pow * 4096.0f + pow;
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
    sfpi::vFloat y = _sfpu_exp_fp32_accurate_<true, true>(frac);
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

    sfpi::vFloat abs_pow = sfpi::abs(sfpi::setsgn(pow, 1));

    // Post-processing: ensure that special values (e.g. 0**0, -1**0.5, ...) are handled correctly
    // Check valid base range
    sfpi::vFloat pow_x = sfpi::min(sfpi::max(sfpi::vFloat(0.5f), abs_pow), 16777216.0f);
    sfpi::vInt pow_bits = sfpi::exman(pow_x, sfpi::MantissaMode::ImplicitOne);
    pow_bits = sfpi::shft(pow_bits, sfpi::exexp(pow_x) + 7, sfpi::ShiftMode::Logical);
    pow_bits = pow_bits + pow_bits;
    y = sfpi::copysgn(y, sfpi::as<sfpi::vFloat>(pow_bits & sfpi::as<sfpi::vInt>(base)));

    v_if(base < 0.0f) {
        // Check for integer power, if it is not then overwrite result with NaN
        v_if((pow_bits + pow_bits) != 0 && abs_pow != 0.0f) {
            y = std::numeric_limits<float>::quiet_NaN();
        }
        v_endif;
    }
    v_endif;

    // setexp/exexp map 0 to log2 = -127, so 0**p evaluates as 2**(-127p).
    // Must follow the negative-base branch: SFPU `<` is sign-bit based, so
    // v_if(base < 0) classes -0 as negative and would overwrite the 0 with the
    // complex-result NaN; -0 is a zero, not a negative base.
    // SFPU `== 0` is bit-exact, so matching -0 needs an abs. Recompute it here
    // rather than reusing abs_base: that live range would span the log2/exp2 body
    // and spills this kernel.
    // Fill 0 for every non-zero exponent, then narrow to the negative ones, which
    // IEEE defines as +inf. v_and tightens the enclosing predicate in place, so it
    // costs one compare where a second flat v_if would also save and restore the
    // lane mask. Gating on |pow| lets both signed zeros keep the mainline's 1.
    // Both fills take their sign from y rather than writing a positive constant, because
    // IEEE keeps the sign of a zero base through an odd integer exponent.
    sfpi::vFloat abs_end = sfpi::abs(base);
    v_if(abs_end == 0.f) {
        v_if(abs_pow != 0.f) {
            y = sfpi::copysgn(sfpi::vFloat(0.0f), y);
            v_and(pow < 0.f);
            y = sfpi::copysgn(sfpi::vFloat(std::numeric_limits<float>::infinity()), y);
        }
        v_endif;
    }
    v_endif;
    v_if(sfpi::is_nan(pow)) { y = std::numeric_limits<float>::quiet_NaN(); }
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

template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_binary_pow(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;
    const uint in0_offset = dst_index_in0 * dst_tile_size_sfpi;
    const uint in1_offset = dst_index_in1 * dst_tile_size_sfpi;
    const uint out_offset = dst_index_out * dst_tile_size_sfpi;

    // Unroll is perf-neutral on BH. Only unrolled in WH_B0.
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[in0_offset];
        sfpi::vFloat in1 = sfpi::dst_reg[in1_offset];

        sfpi::vFloat result = _sfpu_binary_power_<is_fp32_dest_acc_en>(in0, in1);

        sfpi::dst_reg[out_offset] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void sfpu_binary_pow_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstFloatPrgm0 = 1.442695f;
    sfpi::vConstFloatPrgm1 = -127.0f;
}

}  // namespace sfpu
}  // namespace ckernel
