// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>

#include "ckernel.h"
#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_assert.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "sfpi.h"
// After sfpi.h: the Blackhole 21f helpers below use PolynomialEvaluator.
#include "ckernel_sfpu_polyval.h"

namespace ckernel {
namespace sfpu {

// Round-to-nearest-even of a float to its integer value, returning both the rounded float (result)
// and the integer (k_int). Uses the Hacker's Delight 2^23 + 2^22 trick: adding that constant forces
// the fractional bits out, and differencing the raw bit patterns recovers the integer. Only uses
// add/sub plus a bit reinterpret, so it is portable to Quasar (unlike the sign-magnitude round the
// Blackhole kernel interleaves). Valid for |z| < 2^22, which covers exp's reduced argument.
sfpi_inline sfpi::vFloat _sfpu_round_to_nearest_int32_(sfpi::vFloat z, sfpi::vInt& k_int) {
    const sfpi::vFloat c231 = 12582912.0f;  // 2^23 + 2^22
    sfpi::vFloat tmp = z + c231;
    k_int = sfpi::as<sfpi::vInt>(tmp) - sfpi::as<sfpi::vInt>(c231);
    return tmp - c231;
}

/*
 * Branch-free float->int32 conversion for the 21f exp construction, taken from the Blackhole kernel of
 * the same name. Requires 0 <= val < 128.0f and assumes val was already divided by 2^23, so the result
 * is scaled by 2^23 (otherwise the shift would have to be exp - 23).
 *
 * Safe on Quasar: the exponent is non-negative over that range, so the shift amount reads the same in
 * sign-magnitude and two's complement.
 */
sfpi_inline sfpi::vInt _float_to_int32_for_exp_21f_(sfpi::vFloat val) {
    sfpi::vInt exp = sfpi::exexp(val);
    sfpi::vInt man =
        sfpi::exman(val, sfpi::MantissaMode::ImplicitOne);  // get mantissa with implicit bit (man in [1; 2])
    man = sfpi::shft(man, exp, sfpi::ShiftMode::Logical);
    return man;
}

// The Blackhole 21f bf16 exp helpers, carried over verbatim for the ported kernels that call them (i1,
// situ_glu). _float_to_int32_for_exp_21f_ above is the Quasar-safe version they rely on.
/*
 * Unsafe core of BF16 21f exp: skips the xlog2 clamp present in
 * _sfpu_exp_21f_bf16_. The caller MUST ensure `val * (1/ln2) + 127`
 * stays in [0, 256) (roughly val ∈ [-88.0, 88.7]) — otherwise the
 * implicit float→int conversion in _float_to_int32_for_exp_21f_ can
 * wrap and produce garbage.
 *
 * Use this variant when the caller has already clamped its input (e.g. i1's
 * asymptotic path operates on |x| ∈ [10, 88.5]).
 *
 * @param val The input value, must be in the safe range described above.
 * @return sfpi::vFloat Result of exp(val), 21-bit accuracy (~3 FP32 ULP).
 */
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_21f_bf16_unsafe_(sfpi::vFloat val) {
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    sfpi::vFloat xlog2 = (val * ONE_LN2 + 127.f);

    sfpi::vFloat z = sfpi::as<sfpi::vFloat>(_float_to_int32_for_exp_21f_(xlog2));

    sfpi::vInt exponential_part =
        sfpi::exexp(z, sfpi::ExponentMode::Biased);  // Extract exponent ( = 2**(integer part of val/ln2))
    sfpi::vMag fractional_part = sfpi::exman(z);     // Extract mantissa ( = leftover part, in [0; 1])

    sfpi::vFloat frac = sfpi::convert<sfpi::vFloat>(fractional_part, sfpi::RoundMode::Nearest);

    // To refine approximation of 2**(x_f), we use an approximation of 2**x on [0; 2^23]
    // This uses a 2nd degree polynomial adjustment of the fractional part
    frac = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f);

    // Recombined exponent and mantissa: this is equivalent to 2**(x_i) * 2**(x_f)
    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);

    if constexpr (!is_fp32_dest_acc_en) {
        // LRegs work on float32 data. If DST is bfloat16 then SFPSTORE will truncate it.
        // This can reduce accuracy: for instance, 9**2 = 80.8 gets round to 80.5
        // rather than 81 (which would have been correct).
        // To avoid this issue, we explicitly convert to bfloat16 using round-to-nearest.
        y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
    }

    return y;
}

/*
 * This function implements the exponential function using a polynomial approximation algorithm
 * based on "Simple Multiple Precision Algorithms for Exponential Functions [Tips & Tricks]"
 * by Moroz et al. 2022 (https://doi.org/10.1109/MSP.2022.3157460).
 * More specifically, it is the implementation of the `exp_21f` algorithm described in Section 5
 *
 * @param val The input value (sfpi::vFloat vector), can be any floating point number
 *
 * @return sfpi::vFloat Result of exp(val)
 *
 * @see Moroz et al. 2022 - "Simple Multiple Precision Algorithms for Exponential Functions"
 *      ( https://doi.org/10.1109/MSP.2022.3157460 )
 */
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_21f_bf16_(sfpi::vFloat val) {
    // This function computes exp(x) by leveraging mathematic properties of exp(x):
    // That is, exp(x) = 2**(x / ln2) = 2**(x_i) * 2**(x_f) where
    // - z_i = trunc(x / ln2) (integer part)
    // - z_f = x/ln2 - trunc(x/ln2) (fractional part)
    //
    // The paper relies on the following formula (c.f. Section 2 and 3 of paper):
    // z = (bias + x * factor * N_m); where:
    // factor = log(2) * 2^23
    // bias = 127 * 2^23
    // Fundamentally, the formula in the paper computes
    // z = val * log(2) * 2^23 + 127 * 2^23
    // This formula prepares for the computation of exp(x) = 2^(x/log(2))
    //
    // In our case, we will let the multiplication by 2^23 be done implicitly in _float_to_int32_exp21f_ function
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    sfpi::vFloat xlog2 = (val * ONE_LN2 + 127.f);

    // Intermediary values can overflow in xlog2 is outside of [0, 256[ which leads to invalid results instead of 0
    // (when input < -88.5) and +inf (when input > 88.5)
    // To avoid this, we clamp xlog2 to [0, 255]
    // (thresholds values are rounded to bf16, as it does not change result but only requires one SFPLOADI vs. two)
    xlog2 = sfpi::clamp(xlog2, 0.0f, 255.0f);

    sfpi::vFloat z = sfpi::as<sfpi::vFloat>(_float_to_int32_for_exp_21f_(xlog2));

    sfpi::vInt exponential_part =
        exexp(z, sfpi::ExponentMode::Biased);     // Extract exponent ( = 2**(integer part of val/ln2))
    sfpi::vMag fractional_part = sfpi::exman(z);  // Extract mantissa ( = leftover part, in [0; 1])

    sfpi::vFloat frac = sfpi::convert<sfpi::vFloat>(fractional_part, sfpi::RoundMode::Nearest);

    // To refine approximation of 2**(x_f), we use an approximation of 2**x on [0; 2^23]
    // This uses a 2nd degree polynomial adjustment of the fractional part
    frac = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f);

    // Recombined exponent and mantissa: this is equivalent to 2**(x_i) * 2**(x_f)
    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);
    // Quasar: saturated inputs leave exponent field 255 under a nonzero mantissa, i.e. a NaN rather than
    // +inf. Blackhole's final bf16 rounding (SFPSTOCHRND) turns that NaN into +inf; Quasar's SFP_STOCH_RND
    // keeps the NaN, so saturate explicitly.
    v_if(exponential_part >= 255) { y = std::numeric_limits<float>::infinity(); }
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

/*
 * The _sfpu_exp_fp32_accurate_ code is derived from code by Norbert Juffa.
 *
 * Copyright (c) 2015-2021, Norbert Juffa
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
// Non-finite behaviour of this path:
//   +NaN -> NaN    -NaN -> 0    +Inf -> +Inf    -Inf -> 0
// -NaN diverges from Blackhole, which returns NaN for either sign: a negative-signed NaN drives i
// negative, so the lane lands in the underflow arm. Derived from simulation of this routine, NOT
// silicon-verified -- no Quasar target available.
// unsafe = true drops the overflow/underflow guards, as the Blackhole template of the same name does;
// only for callers that already bound the argument inside exp's finite range.
template <bool unsafe = false>
sfpi_inline sfpi::vFloat _sfpu_exp_fp32_accurate_(sfpi::vFloat a) {
    sfpi::vInt i;
    sfpi::vFloat f, r, j;

    // j = round(a / ln2) (as a float) and i = the same value as an integer, interleaved with the
    // first coefficient of the polynomial.
    r = 1.37805939e-3f;
    j = _sfpu_round_to_nearest_int32_(1.442695f * a, i);

    // f = a - j*ln2 (two-part Cody-Waite).
    f = j * -6.93145752e-1f + a;
    f = j * -1.42860677e-6f + f;

    // r = exp(f) on [-ln2/2, ln2/2] via a degree-6 minimax polynomial in Horner form.
    r = r * f + 8.37312452e-3f;  // 0x1.125edcp-7
    r = r * f + 4.16695364e-2f;  // 0x1.555b5ap-5
    r = r * f + 1.66664720e-1f;  // 0x1.555450p-3
    r = r * f + 4.99999851e-1f;  // 0x1.fffff6p-2
    r = r * f + 1.0f;
    r = r * f + 1.0f;

    // exp(a) = 2^i * exp(f), applied via the result's biased exponent e. The legal IEEE-754 range
    // is 1..254, and i pushes e outside it for |a| beyond ~88. Writing (i + 127) << 23 directly
    // would wrap into the sign bit there -- Quasar's integer adder is 32-bit two's complement and
    // SHFT discards the high bits (Quasar/Trinity SFPU MAS), so a = -100 yields i + 127 = -17 ->
    // 0xF7800000 = -2^112 rather than ~0. Nothing routes large-magnitude inputs away from this
    // path (calculate_exponential selects purely on EN_32BIT_DEST / APPROXIMATION_MODE).
    //
    // Seed y with the overflow result unpredicated, as the Blackhole source does, and take the
    // exponent path only when e is in range. Seeding from a (not from the polynomial, which is
    // NaN for a = +-Inf because f = j*-ln2 + a is -Inf + Inf) keeps exp(+Inf) = +Inf.
    //
    // NB: the earlier Quasar port bug was elsewhere -- the Blackhole kernel's sign-magnitude
    // rounding (abs(as<vInt>(convert<vSMag16>(x))) + copysgn feeding a two's-complement add),
    // which relies on Blackhole's integer-format behaviour. _sfpu_round_to_nearest_int32_ above
    // replaces that.
    sfpi::vInt e = sfpi::exexp(r, sfpi::ExponentMode::Biased) + i;
    if constexpr (unsafe) {
        return sfpi::setexp(r, e);
    }
    sfpi::vFloat y = a * std::numeric_limits<float>::infinity();
    v_if(e < 255) {
        y = sfpi::setexp(r, e);
        v_if(e < 1) { y = 0.0f; }  // underflow, incl. subnormals
        v_endif;
    }
    v_endif;
    return y;
}

sfpi_inline sfpi::vFloat _sfpu_exp_fp32_accurate_unsafe_(sfpi::vFloat x) { return _sfpu_exp_fp32_accurate_<true>(x); }

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_accurate_(sfpi::vFloat val);

// is_fp32_dest_acc_en == false
template <>
sfpi_inline sfpi::vFloat _sfpu_exp_accurate_<false>(sfpi::vFloat val) {
    return _sfpu_exp_21f_bf16_<false>(val);
}

// is_fp32_dest_acc_en == true
template <>
sfpi_inline sfpi::vFloat _sfpu_exp_accurate_<true>(sfpi::vFloat val) {
    return _sfpu_exp_fp32_accurate_(val);
}

// Calculates EXP over a full tile. Quasar exposes exactly two implementations:
//   - approximate exp via the HW nonlinear lookup table (sfpi::approx_exp), and
//   - full-precision fp32 exp (_sfpu_exp_fp32_accurate_, ported from Blackhole).
// The LUT is ~1 ULP once the result lands in a bf16 Dest, so the accurate path is only worth
// running for a 32-bit Dest in non-approximate mode; every bf16 case (and any explicit approx
// request) uses the LUT. EN_32BIT_DEST (is_fp32_dest_acc_en) selects the accurate path.
template <
    bool APPROXIMATION_MODE,
    bool EN_32BIT_DEST,
    bool SCALE_EN /*maybe_unused*/ = false,
    int ITERATIONS = SFPU_ITERATIONS,
    bool CLAMP_NEGATIVE /*maybe_unused*/ = true>
void calculate_exponential([[maybe_unused]] const std::uint32_t exp_base_scale_factor = p_sfpu::kCONST_1_FP16B) {
    static_assert(SCALE_EN == false, "Non-default SCALE_EN not supported in Quasar exp");
    static_assert(CLAMP_NEGATIVE == true, "Non-default CLAMP_NEGATIVE not supported in Quasar exp");
    LLK_ASSERT(
        exp_base_scale_factor == p_sfpu::kCONST_1_FP16B,
        "Scaling is not supported in the current version of exp on Quasar.");
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];  // load x from dest (SFPLOAD)

        sfpi::vFloat result;
        if constexpr (!EN_32BIT_DEST || APPROXIMATION_MODE) {
            result = sfpi::approx_exp(val);
        } else {
            result = _sfpu_exp_fp32_accurate_(val);
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <
    bool APPROXIMATION_MODE /*maybe_unused*/,
    uint32_t scale /*maybe_unused*/ = 0x3F800000,
    bool CLAMP_NEGATIVE /*maybe_unused*/ = true,
    bool EN_32BIT_DEST /*maybe_unused*/>
void exp_init() {
    static_assert(scale == 0x3F800000, "Non-default scale not supported in Quasar exp");
    static_assert(CLAMP_NEGATIVE == true, "Non-default CLAMP_NEGATIVE not supported in Quasar exp");
    llk_math_eltwise_unary_sfpu_init<SfpuType::exponential>();
}

}  // namespace sfpu
}  // namespace ckernel
