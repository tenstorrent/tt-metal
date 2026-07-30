// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

#include "ckernel_addrmod.h"
#include "ckernel_ops.h"
// clang-format off: sfpi_inline must be defined before ckernel_sfpu_polyval.h
#include "sfpi.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpu/ckernel_sfpu_polyval.h"
// clang-format on
#include "ckernel_sfpu_recip.h"
#include "lltt.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel {
namespace sfpu {

/*
 * _float_to_int32_positive_ use branch to handle special cases
 * With exp21f function, some of these cases never happen (e.g. negative exponent, overflow)
 * This allow for a branch free (and much smaller algorithm) to compute integer value
 *
 * The constraint on `val` is: 0 <= val < 128.0f
 * Note: Unlike _float_to_int32_positive, this function assumes that
 * value has been been divided by 2^23. Output value will be scaled by 2^23 compared to 'val'.
 * If that was not the case, we would have had to shift by `exp - 23` instead of `exp`
 * This saves 1 SFPADDI instruction.
 */
sfpi_inline sfpi::vInt _float_to_int32_for_exp_21f_(sfpi::vFloat val) {
    sfpi::vInt exp = sfpi::exexp(val);
    sfpi::vInt man =
        sfpi::exman(val, sfpi::MantissaMode::ImplicitOne);  // get mantissa with implicit bit (man in [1; 2])
    man = sfpi::shft(man, exp, sfpi::ShiftMode::Logical);
    return man;
}

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
        sfpi::exexp(z, sfpi::ExponentMode::Biased);    // Extract exponent ( = 2**(integer part of val/ln2))
    sfpi::vMag fractional_part = sfpi::exman(z);       // Extract mantissa ( = leftover part, in [0; 1])

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
 * Implementation of _sfpu_exp_21f_bf16_ (same algorithm) with TTI intrinsics
 * This implementation is faster, and give comparable accuracy as _sfpu_exp_21f_bf16_
 * (~< 1 ULP).
 *
 * Requires _init_exponential_tti_bf16_() to have been called to configure
 * ADDR_MOD_6 (dest auto-increment by 2 on SFPSTORE) and to load:
 *   - LREG12 = 1/ln2 (sfpi::vConstFloatPrgm0)
 *   - LREG13 = c2    (sfpi::vConstFloatPrgm1)  — poly coeff 4.791750e-15f
 */
template <bool SCALE_EN, bool is_fp32_dest_acc_en, bool CLAMP_NEGATIVE, int ITERATIONS>
inline void _sfpu_exp_21f_bf16_tti_(const std::uint16_t exp_base_scale_factor) {
    // Iteration-invariant constants. Loaded once before the loop.
    //
    //   LREG5 = 127.0f                      (bias term in z = x/ln2 + 127)
    //   LREG6 = 7.839635491371155e-08f      (poly coeff c1)
    //   LREG7 = 1.0017248f                  (poly coeff c0)
    //   LREG12 = 1/ln2                      (programmable, set in init)
    //   LREG13 = 4.791750143340323e-15f     (poly coeff c2; programmable, set in init)
    //
    // In-loop scratch:
    //   LREG0 = val → integer-part work
    //   LREG1 = 255 (loaded inside loop) → exexp result → frac (int) → frac (float) → poly result
    //   LREG2 = poly accumulator
    //   LREG3 = xlog2 (preserved through int-part work) → mask
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_FLOATB, 0x42fe);

    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_UPPER, 0x33a8);
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_LOWER, 0x5ada);

    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATA, 0x3c02);

    // Number of instructions in one iteration of the loop body. Used by
    // TTI_REPLAY/record and TTI_REPLAY/replay below; MUST match exactly the count of
    // TTI_ instructions emitted between the TTI_REPLAY/record call and the
    // replay loop, or the replay buffer will misalign.
    //
    //   Base body:                                                15
    //     SFPLOAD, SFPMAD, SFPLOADI(255), SFPSWAP,
    //     SFPEXEXP, SFPEXMAN8, SFPSHFT, SFPEXMAN9, SFPCAST,
    //     SFPMAD poly1, SFPGT mask, SFPMAD poly2,
    //     SFPAND, SFPSETEXP, SFPSTORE.
    //   + SCALE_EN ? 1 : 0                  (SFPMULI scale)
    //   + is_fp32_dest_acc_en ? 0 : 1       (SFP_STOCH_RND fp32→bf16)
    constexpr int BODY_LEN = 15 + (SCALE_EN ? 1 : 0) + (is_fp32_dest_acc_en ? 0 : 1);

    // Record the loop body into replay buffer slot 0 the first time
    // through. Subsequent iterations replay the recorded sequence, which
    // shrinks the unrolled kernel binary from ~ITERATIONS*BODY_LEN
    // instructions down to BODY_LEN + (ITERATIONS - 1) replays.
    //
    // Per-element runtime is unchanged: the recorded instructions execute
    // exactly as if they had been issued inline, and dst_reg is advanced
    // by ADDR_MOD_6 inside the body so each replay walks to the next
    // element correctly.
    //
    // The accurate path (APPROXIMATION_MODE=false) does not otherwise use
    // the replay buffer, so slot 0 is free here. Callers that mix this
    // function with other replay-buffer clients should ensure they don't
    // require slot 0 to survive across the call.
    TTI_REPLAY(0, BODY_LEN, 1, 1);  // record

    // val = sfpi::dst_reg[0]
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);

    if constexpr (SCALE_EN) {
        // Multiply LREG0 by the BF16 scale immediate in-place.
        TTI_SFPMULI(exp_base_scale_factor, p_sfpu::LREG0, 0);
    }

    // xlog2 = val * (1/ln2) + 127.0f, into LREG3 (preserved past int-part work).
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG12, p_sfpu::LREG5, p_sfpu::LREG3, 0);

    // LREG1 = 255.0f. Slots into the SFPMAD's 2-cycle latency window.
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x437f);

    // Upper clamp. SFPSWAP (mode VEC_MIN_MAX = "max into lreg_dest"):
    //   LREG1 = max(255, xlog2), LREG3 = min(255, xlog2).
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);

    // _float_to_int32_for_exp_21f_: shift mantissa left by exp-bias bits.
    // Reads xlog2 from LREG3, leaves int_part in LREG0 (LREG0 freed of val).
    TTI_SFPEXEXP(0, p_sfpu::LREG3, p_sfpu::LREG1, 0);  // LREG1 = exexp(xlog2)
    TTI_SFPEXMAN(0, p_sfpu::LREG3, p_sfpu::LREG0, 0);  // LREG0 = exman8(xlog2)
    TTI_SFPSHFT(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);   // LREG0 <<= LREG1   (int_part)

    // Extract fractional part (sfpi::exman9 with PAD9). LREG0 still holds
    // the integer-part-as-float-encoding which feeds SETEXP later.
    TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPEXMAN_MOD1_PAD9);

    // frac = convert<vFloat>(fractional_part, RoundMode::Nearest)
    constexpr unsigned SFPCAST_MOD1_SM32_TO_FP32_RNE = 0;
    TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, SFPCAST_MOD1_SM32_TO_FP32_RNE);

    // Polynomial refinement of 2^x_f on [0, 1] in Horner form:
    //   frac = c0 + frac * (c1 + frac * c2)
    //        = 1.0017248 + frac * (7.84e-08 + frac * 4.79e-15)
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG13, p_sfpu::LREG6, p_sfpu::LREG2, 0);

    // Negative-input handling: instead of clamping xlog2 with a max(0, ·)
    // SFPSWAP, we build a mask with (SFPGT) and mask negative value using
    // (SFPAND) below.
    // This avoids SFPLOADI + SFPSWAP and introduced instructions
    // can be interleaved with SFPMAD to hide their latency.
    constexpr unsigned SFPGT_MOD1_SET_VD = 8;
    TTI_SFPGT(0, p_sfpu::LCONST_0, p_sfpu::LREG3, SFPGT_MOD1_SET_VD);

    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG7, p_sfpu::LREG1, 0);

    // Apply the mask to the integer part *before* SETEXP: for lanes with
    // xlog2 <= 0 the mask is 0, zeroing the int_part. The subsequent
    // SETEXP then produces a bf16 subnormal that flushes to 0, matching
    // the scalar max(xlog2, 0) behavior on finite inputs.
    constexpr unsigned SFPAND_MOD1_USE_VB = 1;
    TTI_SFPAND(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG0, SFPAND_MOD1_USE_VB);

    // y = setexp(frac, masked_int_part) — recombine 2^x_i * 2^x_f.
    constexpr unsigned SFPSETEXP_MOD1_ARG_EXPONENT = 2;
    TTI_SFPSETEXP(0, p_sfpu::LREG1, p_sfpu::LREG0, SFPSETEXP_MOD1_ARG_EXPONENT);

    if constexpr (!is_fp32_dest_acc_en) {
        // Round float32 -> bfloat16 using round-to-nearest before
        // SFPSTORE truncates. Avoids ULP loss on values like 9*9 = 80.8.
        TTI_SFP_STOCH_RND(
            sfpi::SFPSTOCHRND_RND_EVEN,
            0,
            p_sfpu::LREG0,
            p_sfpu::LREG0,
            p_sfpu::LREG0,
            sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    }

    // sfpi::dst_reg[0] = y; sfpi::dst_reg++;
    // (ADDR_MOD_6 increments dest by 2 on store.)
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);

#pragma GCC unroll 8
    for (std::uint32_t i = 1; i < ITERATIONS; i++) {
        // Replay the recorded body for the remaining ITERATIONS - 1 elements.
        // Each replay is one REPLAY-equivalent issue; the body executes as
        // if it had been issued inline, dst_reg advancing via ADDR_MOD_6.
        TTI_REPLAY(0, BODY_LEN, 0, 0);
    }
}

// Utility function to round a float to a 32-bit integer while also calculating the
// integer part of the rounded value
sfpi_inline sfpi::vFloat _sfpu_round_to_nearest_int32_(sfpi::vFloat z, sfpi::vInt& k_int) {
    // From Hacker's Delight: round-to-nearest method
    // float -> int32 (round to nearest even): n = (x + float(c231)) - int32(c231)
    // round-to-nearest: n = (x + float(c231)) - float(c231)
    // where c231 = 0x4B400000 (2^23 + 2^22)
    const sfpi::vFloat c231 = Converter::as_float(0x4B400000U);  // 2^23 + 2^22

    sfpi::vFloat tmp = z + c231;
    sfpi::vFloat k = tmp - c231;
    k_int = sfpi::as<sfpi::vInt>(tmp) - sfpi::as<sfpi::vInt>(c231);

    return k;
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

template <bool unsafe = false>
sfpi_inline sfpi::vFloat _sfpu_exp_fp32_accurate_(sfpi::vFloat a) {
    sfpi::vInt i, e;
    sfpi::vFloat f, r, j, y;
    sfpi::vSMag16 sm;

    // j = round(a / ln2)
    // interleaved with first coefficient of polynomial
    j = 1.442695f * a;
    r = 1.37805939e-3f;
    sm = sfpi::convert<sfpi::vSMag16>(j, sfpi::RoundMode::Nearest);
    j = sfpi::convert<sfpi::vFloat>(sm, sfpi::RoundMode::Nearest);

    // f = a - i*j (two-part cody-waite)
    f = j * -6.93145752e-1f + a;
    f = j * -1.42860677e-6f + f;

    // approximate r = exp(f) on [-ln2/2, ln2/2]
    // interleaved with conversion of i from sign-mag to two's complement via abs and copysgn
    r = r * f + 8.37312452e-3f;  // 0x1.125edcp-7
    r = r * f + 4.16695364e-2f;  // 0x1.555b5ap-5
    r = r * f + 1.66664720e-1f;  // 0x1.555450p-3
    r = r * f + 4.99999851e-1f;  // 0x1.fffff6p-2
    i = sfpi::abs(sfpi::as<sfpi::vInt>(sm));
    y = r * f + 1.0f;
    i = sfpi::as<sfpi::vInt>(sfpi::copysgn(sfpi::as<sfpi::vFloat>(i), j));
    r = y * f + 1.0f;

    if constexpr (unsafe) {
        // y = 2**i * r
        e = sfpi::exexp(r, sfpi::ExponentMode::Biased) + i;
        y = sfpi::setexp(r, e);
    } else {
        // overflow: y = infinity or NaN
        y *= std::numeric_limits<float>::infinity();

        e = sfpi::exexp(r, sfpi::ExponentMode::Biased) + i;
        // if e < 255
        v_block {
            sfpi::vInt e_lt_255 = __builtin_rvtt_sfpiadd_i(e.get(), -255, sfpi::SFPIADD_MOD1_CC_LT0);

            // y = 2**i * r
            y = sfpi::setexp(r, e);

            // if e < 1
            v_if(e_lt_255 < -254) {
                // underflow, including subnormals
                y = 0.0f;
            }
            v_endif;
        }
        v_endblock;
    }

    return y;
}

sfpi_inline sfpi::vFloat _sfpu_exp_fp32_accurate_unsafe_(sfpi::vFloat x) {
    return _sfpu_exp_fp32_accurate_<true>(x);
}

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

sfpi_inline sfpi::vFloat _sfpu_exp_(sfpi::vFloat val) {
    // If exponent is > -1 extract it and replace with -1
    sfpi::vInt exp = exexp(val);
    v_if(exp >= 0) { val = setexp(val, 126); }
    v_endif;

    // Run series in Horner form
    sfpi::vFloat tmp = val * 0.8373f + sfpi::sFloat16b(0.863281f);
    val = val * tmp + 1.0f;

    v_if(exp >= 0) {
        val = val * val;
        for (int s_iter = 0; s_iter < 7; s_iter++) {
            exp = exp - 1;
            // Narrow predication on each loop
            v_and(exp >= 0);
            val = val * val;
        }
    }
    v_endif;

    return val;
}

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat _calculate_exponential_body_(sfpi::vFloat in) {
    sfpi::vFloat out;

    if constexpr (APPROXIMATION_MODE) {
        constexpr int FRAC_BITS = 3;
        constexpr std::uint32_t SP_BIAS = 127 << FRAC_BITS;

        // * by 1/ln2 and add convert to 7.3 FxP format
        sfpi::vFloat vConstLn2Recip = sfpi::vConstFloatPrgm0;
        sfpi::vFloat conv = in * vConstLn2Recip;

        // Clear exp bits
        sfpi::vInt c23_73 = p_exp::C23_73;
        sfpi::vInt tmp = sfpi::as<sfpi::vInt>(conv) - c23_73;

        // Add bias
        tmp += SP_BIAS;

        // SHL to move integer bits to exponent
        out = sfpi::as<sfpi::vFloat>(tmp << (10 - FRAC_BITS));
    } else {
        // Force sign to 0 (make number positive)
        out = _sfpu_exp_(sfpi::setsgn(in, 0));

        v_if(in < 0) { out = sfpu_reciprocal_iter<2>(out); }
        v_endif;
    }

    return out;
}

template <bool SCALE_EN, bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _ckernel_sfpu_exp_accurate_(sfpi::vFloat val, const std::uint32_t exp_base_scale_factor) {
    if constexpr (SCALE_EN) {
        val = val * sfpi::sFloat16b(exp_base_scale_factor);
    }
    sfpi::vFloat result = _sfpu_exp_accurate_<is_fp32_dest_acc_en>(val);
    return result;
}

template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en,
    bool SCALE_EN = false,
    int ITERATIONS = 8,
    bool CLAMP_NEGATIVE = true>
void calculate_exponential(const uint exp_base_scale_factor = p_sfpu::kCONST_1_FP16B) {
    if constexpr (!APPROXIMATION_MODE) {
        if constexpr (!is_fp32_dest_acc_en) {
            // bfloat16-accurate path: hand-tuned TTI exp_21f kernel.
            // CLAMP_NEGATIVE is forwarded for API symmetry with the WH kernel,
            // but on BH the negative-input handling is applied unconditionally
            // because the SFPGT mask hides in an SFPMAD latency window and is
            // effectively free (no other instruction to interleave with SFPMAD).
            _sfpu_exp_21f_bf16_tti_<SCALE_EN, is_fp32_dest_acc_en, CLAMP_NEGATIVE, ITERATIONS>(exp_base_scale_factor);
        } else {
            for (int d = 0; d < ITERATIONS; d++) {
                sfpi::vFloat val = sfpi::dst_reg[0];
                sfpi::dst_reg[0] =
                    _ckernel_sfpu_exp_accurate_<SCALE_EN, is_fp32_dest_acc_en>(val, exp_base_scale_factor);
                sfpi::dst_reg++;
            }
        }
    } else if constexpr (APPROXIMATION_MODE && CLAMP_NEGATIVE) {
#ifdef DISABLE_SFPLOADMACRO
        for (int d = 0; d < ITERATIONS; d++) {
            TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);
            TTI_SFPSWAP(0, p_sfpu::LREG14, p_sfpu::LREG0, 9);
            TTI_SFPMAD(p_sfpu::LREG12, p_sfpu::LREG0, p_sfpu::LREG13, p_sfpu::LREG0, 0);
            TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT16);
            TTI_SFPSHFT(15, p_sfpu::LREG0, p_sfpu::LREG0, 1);
            TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);
            sfpi::dst_reg++;
        }
#else
        // Code below is hand-unrolled for 8 iterations
        // so it doesn't respect ITERATIONS. TODO: tt-llk#1486
        // static_assert(ITERATIONS == 8);

        // Sanitize the input values by loading from DEST, comparing against the value -88.5, and if the input value is
        // more negative than that, swap the input value with -88.5 and store back to DEST
        //  - in other words, after the sanitize step, the values in DEST will be in the range {-88.5 , +inf}

        // Macro Sequence Register 1 configured to read back in the original values from dest, sanitize them to a range
        // we can handle, and then store them back to dest
        //  LD     : bring in the original value from DEST (y)
        //  MAD    : unused
        //  ROUND  : unused
        //  SIMPLE : SWAP the larger value of y and -88.5 into the LREG
        //  STORE  : store the sanitized value back to dest
        TTI_SFPLOADMACRO(
            4,
            0,
            ADDR_MOD_7,
            0);      // MACRO Sequence Register 1: LD, SWAP, STORE - uses LREG[0] for loaded value - Dest offset  0 is
                     // targeting the even columns for rows   3: 0
        TTI_SFPNOP;  // NOP is necessary because the SWAP operation takes 2 cycles and unfortunately is not pipelined
        TTI_SFPLOADMACRO(
            5,
            0,
            ADDR_MOD_7,
            2);  // MACRO Sequence Register 1: LD, SWAP, STORE - uses LREG[1] for loaded value - Dest offset  2 is
                 // targeting the odd  columns for rows   3: 0
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(
            6,
            0,
            ADDR_MOD_7,
            4);  // MACRO Sequence Register 1: LD, SWAP, STORE - uses LREG[2] for loaded value - Dest offset  4 is
                 // targeting the even columns for rows   7: 4
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(
            7,
            0,
            ADDR_MOD_7,
            6);  // MACRO Sequence Register 1: LD, SWAP, STORE - uses LREG[3] for loaded value - Dest offset  6 is
                 // targeting the odd  columns for rows   7: 4
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(
            4,
            0,
            ADDR_MOD_7,
            8);  // MACRO Sequence Register 1: LD, SWAP, STORE - uses LREG[0] for loaded value - Dest offset  8 is
                 // targeting the even columns for rows  11: 8
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(
            5,
            0,
            ADDR_MOD_7,
            10);  // MACRO Sequence Register 1: LD, SWAP, STORE - uses LREG[1] for loaded value - Dest offset 10 is
                  // targeting the even columns for rows  11: 8
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(
            6,
            0,
            ADDR_MOD_7,
            12);  // MACRO Sequence Register 1: LD, SWAP, STORE - uses LREG[2] for loaded value - Dest offset 12 is
                  // targeting the odd  columns for rows  15:12
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(
            7,
            0,
            ADDR_MOD_7,
            14);  // MACRO Sequence Register 1: LD, SWAP, STORE - uses LREG[3] for loaded value - Dest offset 14 is
                  // targeting the even columns for rows  15:12
        // NOP not needed in this spot because the next LoadMacro is a computational macro which doesn't immediately use
        // the SIMPLE unit

        // Macro Sequence Register 0 configured to read back in the sanitized values and calculate the approximate
        // exponential value
        //  LD     : the sanitized value from DEST (y)
        //  MAD    : compute (A * y) + (B-C)  , where A = (2^8)/ln(2) , B = 127 * (2^8) , C = Adjustment parameter of
        //  roughly 11.2 to minimize error ROUND  : convert the MAD result from FP32 to a 16-bit unsigned integer using
        //  stochastic rounding SIMPLE : shift the 16-bit integer to the left by 15 bits to place the MSB of the
        //  computed value into the MSB of the exponent bits of the fp32 format STORE  : store the shifted value back to
        //  dest
        TTI_SFPLOADMACRO(0, 0, ADDR_MOD_7, 0);  // MACRO Sequence Register 0: LD, MAD, ROUND, SHIFT and STORE - uses
                                                // LREG[0] for loading and intermediate results
                                                // - Dest offset  0 is targeting the even columns for rows   3: 0
        TTI_SFPLOADMACRO(1, 0, ADDR_MOD_7, 2);  // MACRO Sequence Register 0: LD, MAD, ROUND, SHIFT and STORE - uses
                                                // LREG[1] for loading and intermediate results
                                                // - Dest offset  2 is targeting the odd  columns for rows   3: 0
        TTI_SFPLOADMACRO(2, 0, ADDR_MOD_7, 4);  // MACRO Sequence Register 0: LD, MAD, ROUND, SHIFT and STORE - uses
                                                // LREG[2] for loading and intermediate results
                                                // - Dest offset  4 is targeting the even columns for rows   7: 4
        TTI_SFPLOADMACRO(3, 0, ADDR_MOD_7, 6);  // MACRO Sequence Register 0: LD, MAD, ROUND, SHIFT and STORE - uses
                                                // LREG[3] for loading and intermediate results
                                                // - Dest offset  6 is targeting the odd  columns for rows   7: 4
        TTI_SFPLOADMACRO(0, 0, ADDR_MOD_7, 8);  // MACRO Sequence Register 0: LD, MAD, ROUND, SHIFT and STORE - uses
                                                // LREG[0] for loading and intermediate results
                                                // - Dest offset  8 is targeting the even columns for rows  11: 8
        TTI_SFPLOADMACRO(
            1,
            0,
            ADDR_MOD_7,
            10);  // MACRO Sequence Register 0: LD, MAD, ROUND, SHIFT and STORE - uses LREG[1] for loading and
                  // intermediate results - Dest offset 10 is targeting the even columns for rows  11: 8
        TTI_SFPLOADMACRO(
            2,
            0,
            ADDR_MOD_7,
            12);  // MACRO Sequence Register 0: LD, MAD, ROUND, SHIFT and STORE - uses LREG[2] for loading and
                  // intermediate results - Dest offset 12 is targeting the odd  columns for rows  15:12
        TTI_SFPLOADMACRO(
            3,
            0,
            ADDR_MOD_7,
            14);  // MACRO Sequence Register 0: LD, MAD, ROUND, SHIFT and STORE - uses LREG[3] for loading and
                  // intermediate results - Dest offset 14 is targeting the even columns for rows  15:12
        // NOP needed to allow time for the final Computation Loadmacro to complete before returning to the Sanitation
        // Loadmacro at the top for the next iteration
        //  - to be completely safe, use 3 NOP; in practice 1 seems to be enough, probably because the overhead of the
        //  DEST INCRW stuff introduces 2 cycles of delay
        TTI_SFPNOP;
        // TTI_SFPNOP;
        // TTI_SFPNOP;
#endif
    }
#ifdef DISABLE_SFPLOADMACRO
    else if constexpr (APPROXIMATION_MODE) {
        for (int d = 0; d < ITERATIONS; d++) {
            TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);
            TTI_SFPMAD(p_sfpu::LREG12, p_sfpu::LREG0, p_sfpu::LREG13, p_sfpu::LREG0, 0);
            TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_INT16);
            TTI_SFPSHFT2(p_sfpu::LREG0, p_sfpu::LREG14, p_sfpu::LREG1, 5);  // lreg[1] = lreg[0] << 15
            TTI_SFPSETSGN(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);  // lreg[0] preserves sign, copies e/m from lreg[1]
            TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);
            sfpi::dst_reg++;
        }
    }
#else
    else if constexpr (APPROXIMATION_MODE && ITERATIONS == 8) {
        // =======================================================================
        // 8-element version using replay buffer.
        // Total: ~20 cycles for 8 elements = 2.5 cycles/element
        //
        // Uses 1 replay of the 16-instruction pattern.
        // First 2 SHFT2s are dummy (timing placeholders), then 6 real SHFT2s.
        // Drain phase handles final 2 SHFT2s.
        // =======================================================================

        // Configure ADDR_MOD_7 for auto-increment (dest += 2 per LOADMACRO).
        addr_mod_t{
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},
        }
            .set(ADDR_MOD_7);

        // Single replay of 16 instructions = 8 LM + 8 SHFT2 (2 dummy + 6 real).
        lltt::replay(0, 16);

        // Drain: SHFT2[6-7].
        TTI_SFPNOP;
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG14, p_sfpu::LREG4, 5);  // SHFT2[6]
        TTI_SFPNOP;
        TTI_SFPSHFT2(p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG4, 5);  // SHFT2[7]
        TTI_SFPNOP;
        TTI_SFPNOP;
    } else if constexpr (APPROXIMATION_MODE && ITERATIONS == 32) {
        // =======================================================================
        // 32-element version using replay buffer.
        // Total: ~68 cycles for 32 elements = 2.125 cycles/element
        //
        // Uses 2 replays of the 32-instruction pattern.
        // First 2 SHFT2s are dummy (timing placeholders), then 30 real SHFT2s.
        // Drain phase handles final 2 SHFT2s.
        // =======================================================================

        // Configure ADDR_MOD_7 for auto-increment (dest += 2 per LOADMACRO).
        addr_mod_t{
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},
        }
            .set(ADDR_MOD_7);

        // 2 replays of 32 instructions = 32 LM + 32 SHFT2 (2 dummy + 30 real).
        lltt::replay(0, 32);
        lltt::replay(0, 32);

        // Drain: SHFT2[30-31].
        TTI_SFPNOP;
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG14, p_sfpu::LREG4, 5);  // SHFT2[30]
        TTI_SFPNOP;
        TTI_SFPSHFT2(p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG4, 5);  // SHFT2[31]
        TTI_SFPNOP;
        TTI_SFPNOP;
    } else if constexpr (APPROXIMATION_MODE) {
        static_assert(
            ITERATIONS == 8 || ITERATIONS == 32, "This version of exponential only supports 8 or 32 iterations.");
    }
#endif
}

constexpr auto bits = [](float x) constexpr { return __builtin_bit_cast(std::uint32_t, x); };
constexpr auto lo16 = [](float x) constexpr { return static_cast<std::uint16_t>(bits(x) & 0xFFFFu); };
constexpr auto hi16 = [](float x) constexpr { return static_cast<std::uint16_t>(bits(x) >> 16); };

template <
    bool APPROXIMATION_MODE,
    uint32_t scale = 0x3F800000,
    bool CLAMP_NEGATIVE = true,
    bool is_fp32_dest_acc_en = false>
void exp_init() {
    // Common SFPU init inlined (SFPU config register + ADDR_MOD_7 + counter reset), then the op-specific
    // exp setup below -- one self-contained init, no separate shared-common-init call. Same functionality as
    // _llk_math_eltwise_unary_sfpu_init_<exponential>() (exp uses only ADDR_MOD_7, no op-specific ADDR_MOD_6).
    sfpu::_init_sfpu_config_reg();
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    math::reset_counters(p_setrwc::SET_ABD_F);
    if constexpr (APPROXIMATION_MODE && CLAMP_NEGATIVE) {
        // Algorithm is adapted from:
        //      A Fast, Compact Approximation of the Exponential Function
        //      Nicol N. Schraudolph
        //      IDSIA, Lugano, Switzerland

        // First, set up constant values which are needed for the computation
        //      We will first sanitize the input values (y) to be in the range that won't cause underflow, which for our
        //      hardware means we need to limit negative values to be greater than or equal to -88.5 The computation
        //      that is needed is (A * y) + (B - C) , where A = (2^8)/ln(2) , B = 127 * (2^8) , C = Adjustment parameter
        //      of roughly 11.2 to minimize error
        //          - NOTE: we would like to be able to use 2^23 instead of 2^8 and compute a 32-bit quantity, but our
        //          hardware only supports rounding FP32 into a 16-bit integer, so we use 2^8 and then shift left by 15
        //          bits after rounding
        //      So we will set up the following constants:
        //          LREG[14] =       =    -88.5               = 0xc2b10000
        //          LREG[12] = A     =    369.329925537109375 = 0x43b8aa3b
        //          LREG[13] = (B-C) =  32500.818359375       = 0x46fde9a3

        constexpr float LN2_RECIP = 1.4426950408889634f;
        constexpr float A = 256.0f * LN2_RECIP;
        constexpr float B_minus_C = 32500.818359375f;
        constexpr float THRESHOLD = -88.5f;

        constexpr float scale_fp32 = __builtin_bit_cast(float, scale);

        constexpr float A_scaled = A * scale_fp32;
        constexpr float THRESHOLD_scaled = THRESHOLD / scale_fp32;

        TTI_SFPLOADI(0, 0xA, lo16(THRESHOLD_scaled));
        TTI_SFPLOADI(0, 0x8, hi16(THRESHOLD_scaled));
        TTI_SFPCONFIG(0, 14, 0);  // SFPCONFIG Dest 14 = LREG[14] =            -88.5               = 0xc2b10000

        TTI_SFPLOADI(0, 0xA, lo16(A_scaled));
        TTI_SFPLOADI(0, 0x8, hi16(A_scaled));
        TTI_SFPCONFIG(0, 12, 0);  // SFPCONFIG Dest 12 = LREG[12] = A     =    369.329925537109375 = 0x43b8aa3b

        TTI_SFPLOADI(0, 0xA, lo16(B_minus_C));
        TTI_SFPLOADI(0, 0x8, hi16(B_minus_C));
        TTI_SFPCONFIG(0, 13, 0);  // SFPCONFIG Dest 13 = LREG[13] = (B-C) =  32500.818359375       = 0x46fde9a3

#ifndef DISABLE_SFPLOADMACRO
        // Next, set up the macro instructions which will be necessary
        //  - for the sanitize function: we will need a SWAP instruction
        //  - for the main computation function: we will need MAD, ROUND, and SHIFT instructions

        // There are two ways to program the macro instruction registers, and this setup leverages both ways
        //  - we can either use the SFPCONFIG flow, by setting up the bits of the instruction into LREG[0] and then
        //  targeting the Macro instruction register
        //  - or we can use the shortcut / backdoor load method which relies on having some illegal destination register
        //  values as part of the instruction

        // Use SFPCONFIG method for the SWAP instruction, since we want the SWAP itself to use a destination register
        // which is not normally a legal value
        //      (we are cheating a bit here, since we only care about one half of the swap and we want to use a constant
        //      for the other half)
        //
        //              imm12 = 0,       lreg_src_c = 0 (will be fed by value loaded from Dest into Loadmacro
        //              lreg_dest),  lreg_dest = LREG[14] = - 88.5, instr_mod1 = 1 swap the values with the larger of
        //              the two ending up in lreg_dest -> but we will use the Loadmacro lreg_dest register as output
        // TTI_SFP_SWAP(0,               0, 14,                            1);
        TTI_SFPLOADI(0, 0xA, 0x00E1);
        TTI_SFPLOADI(0, 0x8, 0x9200);
        TTI_SFPCONFIG(
            0, 0, 0);  // SFPCONFIG Dest 0 = Programmable Macro instruction 0: TTI_SFPSWAP(0, 0, 14, 1); // compare
                       // against LREG[14] (-88.5), and put the larger value into LREG[loadmacro_lreg_dest]
        TTI_SFPNOP;

        // Backdoor load of Macro Instruction 1
        // Dummy version of MAD instruction with lreg_dest = 4'b11_01 = 13 to install into Programmable Macro
        // instruction register 1, which is Macro Instruction Register 5
        TTI_SFPMAD(12, 0, 13, 13, 0);  // MACRO Instruction 1 <--- lreg X = lreg[12] (A) * lreg[0] (y) + lreg[13] (B-C)

        // Backdoor load of Macro Instruction 2
        // ROUND instruction to convert FP32 result into an integer value (int16)
        //                Stochastic = 0,  Imm(Descale),  SrcB(unused),   SrcC(input value),  Lreg_dest = 14 to install
        //                in Programmable Macro Instruction reg 2'b10,  instr_mod1 = 14 to treat input as fp32, output
        //                as unsigned int16, use imm as descale
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 14, 14);  // Round to unsigned Int16

        // Backdoor load of Macro Instruction 3
        // If using the unsigned int rounding mode, then shift by 15; SHL to move integer bits to exponent;
        TTI_SFPSHFT(
            15,
            0,
            15,
            1);  // imm = 15 to shift left by 15 bits; lreg_c = 0 (will use macro reg); lreg_dest = 15 to install in
                 // Programmable Macro Instruction reg 2'b11, which is Macro Instruction Register 7

        // So at this point, we have the following instructions loaded into our macro registers:
        //
        // 00: (no macro instruction, just execute whatever is issued from Tensix) <-- these are fixed / not
        // programmable 01: ( Rsvd                                                            ) <-- these are fixed /
        // not programmable 02: ( NOP                                                             ) <-- these are fixed
        // / not programmable 03: ( SFPSTORE                                                        ) <-- these are
        // fixed / not programmable 04: TTI_SFPSWAP       (0, 0, 11, 1) 05: TTI_SFPMAD        (12, 0, 13, 13, 0) 06:
        // TTI_SFP_STOCH_RND (1, 0, 0, 0, 14, 14) 07: TTI_SFPSHFT       (15,0,15,1)

        // Now we want to set up our two sequences

        // Sequence 1 setup: we want to Load, SWAP, <delay>, Store
        //       Delay slot:                  0     1        2
        //                                                                                                                                                                                                 Use
        //                                                                                                                                                                                                 Loaded  Result          Macro
        //                                                                                                                                                                                                 Value   Value   Delay   Instruction
        //                                                                                                                                                                                                 SRCB    Stage   Slot    Select
        TTI_SFPLOADI(
            0,
            0xA,
            0x0004);  // slot1 : SIMPLE UNIT, want SWAP  instruction which is in macro instruction mux[4], delayed by 0
                      // ; not using staging flop as dest; not using load reg as srcb : 8'b0_______0_______000_____100
                      // = 0x04 slot2 : MAD    UNIT, unused : 8'b0_______0_______000_____000          = 0x00
        TTI_SFPLOADI(
            0, 0x8, 0x1300);  // slot3 : ROUND  UNIT, unused : 8'b0_______0_______000_____000          = 0x00 slot4 :
                              // STORE  UNIT, want STORE instruction which is in macro instruction mux[3], delayed by 2
                              // ; not using staging flop as src ; : 8'b0_______0_______010_____011          = 0x13
        TTI_SFPCONFIG(0, 5, 0);  // SFPCONFIG Dest 5 = Macro Sequence Register 1

        // Sequence 0 setup: we want to Load, MAD, <delay>, ROUND, SHIFT, Store
        //       Delay slot:                  0    1        2      3      4
        //                                                                                                                                                                                                 Use
        //                                                                                                                                                                                                 Loaded  Result          Macro
        //                                                                                                                                                                                                 Value   Value   Delay   Instruction
        //                                                                                                                                                                                                 SRCB    Stage   Slot    Select
        TTI_SFPLOADI(
            0,
            0xA,
            0x85DF);  // slot1 : SIMPLE UNIT, want SHIFT instruction which is in macro instruction mux[7], delayed by 3
                      // ;     using staging flop as dest; using load reg as srcb : 8'b1_______1_______011_____111 =
                      // 0xDF slot2 : MAD    UNIT, want MAD   instruction which is in macro instruction mux[5], delayed
                      // by 0 ; not using staging flop as dest;     using load reg as srcb :
                      // 8'b1_______0_______000_____101 = 0x85
        TTI_SFPLOADI(
            0,
            0x8,
            0x6316);  // slot3 : ROUND  UNIT, want ROUND instruction which is in macro instruction mux[6], delayed by 2
                      // ; not using staging flop as dest; using : 8'b0_______0_______010_____110          = 0x16 slot4
                      // : STORE  UNIT, want STORE instruction which is in macro instruction mux[3], delayed by 4 ;
                      // using staging flop as src ;     using                  : 8'b0_______1_______100_____011 = 0x63
        TTI_SFPCONFIG(0, 4, 0);  // Load it into macro sequence register 0 (destination = 4)

        // Reset LoadMacroConfig[Lane].Misc for all lanes, in case it has been previously set by another use of macros.
        TTI_SFPCONFIG(0, 8, 1);
#endif
    } else if constexpr (APPROXIMATION_MODE) {
        // ===================================================================
        // Based on "A Fast, Compact Approximation of the Exponential Function" by Schraudolph.
        //
        // The Schraudolph algorithm computes exp(x) by exploiting the fact that IEEE 754 floats
        // encode values as 2^(exponent) * (1 + mantissa), where the bit-pattern read as an integer
        // is linear in log2(value). This allows exp(x) to be computed as i = A*x + (B-C)
        // when reinterpreted as float.
        // This implementation adds an explicit sign-setting step (SETSGN) to ensure outputs are
        // negative for inputs below ~-88, where the algorithm would otherwise produce incorrect values.
        // To get a correct result for inputs below ~-88 the output of this function must be ReLU'd.
        // In this implementation, for inputs above 0.72 the output saturates to exp(0.72).
        // Valid input range: [-88, 0.72] with no following ReLU, [-inf, 0.72] with ReLU.
        //
        // Constants:
        //   LREG[12] = A = 256.0 * (1/ln2) = 369.329925537109375
        //   LREG[13] = B - C = 32500.818359375
        //   LREG[14] = 15 (shift amount for SFPSHFT2)
        //
        // Macro Instructions (backdoor loaded):
        //   Macro 5: MAD        - compute i = A * x + (B-C)
        //   Macro 6: STOCHRND   - convert to INT16 (sign-magnitude format)
        //   Macro 7: SETSGN     - restore sign from STOCHRND result to shifted value
        //
        // Macro Sequence Register 0
        //   Slot 1 (Simple): SETSGN @ delay 5, writes to LREG[16] (staging)
        //     - Bit 7 = 1: VB = loadmacro's VD (sign source from STOCHRND result)
        //     - Bit 6 = 1: VD = 16 (staging register, avoids write port conflict)
        //     - Bits 5:3 = 101: delay 5
        //     - Bits 2:0 = 111: macro 7 (SETSGN)
        //     - Encoding: 0b1_1_101_111 = 0xEF
        //
        //   Slot 2 (MAD): MAD @ delay 0
        //     - Bit 7 = 1: use loaded value as VB
        //     - Bit 6 = 0: write to LREG[lreg_dest]
        //     - Bits 5:3 = 000: delay 0
        //     - Bits 2:0 = 101: macro 5 (MAD)
        //     - Encoding: 0b1_0_000_101 = 0x85
        //
        //   Slot 3 (Round): STOCHRND @ delay 3
        //     - Bits 5:3 = 011: delay 3
        //     - Bits 2:0 = 110: macro 6 (STOCHRND)
        //     - Encoding: 0b0_0_011_110 = 0x1E
        //
        //   Slot 4 (Store): STORE @ delay 6, reads from LREG[16]
        //     - Bit 7 = 0: don't preserve VD from instruction
        //     - Bit 6 = 1: read from LREG[16] (staging register)
        //     - Bits 5:3 = 110: delay 6
        //     - Bits 2:0 = 011: macro 3 (STORE)
        //     - Encoding: 0b0_1_110_011 = 0x73
        //
        // =======================================================================

        constexpr float LN2_RECIP = 1.4426950408889634f;
        constexpr float A = 256.0f * LN2_RECIP;
        constexpr float B_minus_C = 32500.818359375f;

        constexpr float scale_fp32 = __builtin_bit_cast(float, scale);
        constexpr float A_scaled = A * scale_fp32;

        // Load constant A into LREG[12]
        TTI_SFPLOADI(0, 0xA, lo16(A_scaled));
        TTI_SFPLOADI(0, 0x8, hi16(A_scaled));
        TTI_SFPCONFIG(0, 12, 0);

        // Load constant (B-C) into LREG[13]
        TTI_SFPLOADI(0, 0xA, lo16(B_minus_C));
        TTI_SFPLOADI(0, 0x8, hi16(B_minus_C));
        TTI_SFPCONFIG(0, 13, 0);

        // Load shift amount (15) into LREG[14] for SFPSHFT2
        // SFPSHFT2 mode 5 reads shift amount from VC register
        TTI_SFPLOADI(0, 0xA, 15);  // Lower 16 bits = 15
        TTI_SFPLOADI(0, 0x8, 0);   // Upper 16 bits = 0
        TTI_SFPCONFIG(0, 14, 0);   // Store in LREG[14]

#ifndef DISABLE_SFPLOADMACRO
        // ===================================================================
        // Program Macro Instructions via Backdoor Load
        // ===================================================================

        // Macro Instruction 1 (slot 5): MAD
        // Computes: LREG[dest] = LREG[12] * LREG[dest] + LREG[13]
        // dest=13 triggers backdoor load to Macro Instruction Register 5
        TTI_SFPMAD(12, 0, 13, 13, 0);

        // Macro Instruction 2 (slot 6): STOCHRND with INT16 mode
        // Converts FP32 to sign-magnitude integer with max magnitude 32767
        // dest=14 triggers backdoor load to Macro Instruction Register 6
        // Mode 7 = FP32_TO_INT16 (keeps sign in bit 31, clamps magnitude to 32767)
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 14, 7);

        // Macro Instruction 3 (slot 7): SETSGN
        // VC=4: reads exp/mantissa from LREG[4] (where discrete SHFT2 writes)
        // VD=15: triggers backdoor load to Macro Instruction Register 7
        // Mod1=0: sign comes from VB (which equals loadmacro's VD after override)
        //
        // When executed via LOADMACRO with sequence bits:
        //   - Bit 7=1: VB = loadmacro's VD (0-3, sign source from STOCHRND)
        //   - Bit 6=1: VD = 16 (staging register for output)
        //   - VC is preserved as 4 (exp/man source from SHFT2 result)
        TTI_SFPSETSGN(0, 4, 15, 0);

        //   Low 16 bits:  Slot2(MAD)=0x85, Slot1(Simple)=0xEF -> 0x85EF
        //   High 16 bits: Slot4(Store)=0x73, Slot3(Round)=0x1E -> 0x731E
        //   - STOCHRND delay 3: 0x1E
        //   - SETSGN delay 5: 0xEF
        //   - STORE delay 6: 0x73
        TTI_SFPLOADI(0, 0xA, 0x85EF);  // Slots 1-2: Simple=0xEF (delay 5), MAD=0x85
        TTI_SFPLOADI(0, 0x8, 0x731E);  // Slots 3-4: Round=0x1E (delay 3), Store=0x73 (delay 6)
        TTI_SFPCONFIG(0, 4, 0);        // Load into Macro Sequence Register 0 (dest=4)

        // Reset LoadMacroConfig[Lane].Misc for all lanes
        // Sets StoreMod0=0 (SRCB), UsesLoadMod0ForStore=0, UnitDelayKind=0xF
        // UnitDelayKind prevents pipeline advancement when not seeing new instructions,
        // avoiding race conditions from dest bank conflicts or other pipeline hiccups.
        TTI_SFPCONFIG(0xF00, 0x8, 0x1);

        // ===================================================================
        // Program Replay Buffer
        // ===================================================================
        // Record 32 instructions (16 LM+SHFT2 pairs) into replay buffer.
        // ADDR_MOD_7 will be configured for auto-increment at replay time.
        //
        // Pattern starts with LM(LREG0) so all LOADMACROs can be replayed.
        // The first two SHFT2s are "dummy" (operate on not-yet-loaded LREGs)
        // but provide correct pipeline timing for subsequent real SHFT2s.
        //
        // LREG pattern cycles every 4 elements:
        //   LM uses:    LREG0, LREG1, LREG2, LREG3, LREG0, ...
        //   SHFT2 uses: LREG2, LREG3, LREG0, LREG1, LREG2, ...
        // ===================================================================

        lltt::record(0, 32);

        // 16 pairs of LM + SHFT2 (LREG pattern repeats every 4 pairs)
        // Pairs 0-1: dummy SHFT2s, Pairs 2-15: real SHFT2s for elements 0-13
        TTI_SFPLOADMACRO(0, 0, ADDR_MOD_7, 0);
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPLOADMACRO(1, 0, ADDR_MOD_7, 0);
        TTI_SFPSHFT2(p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPLOADMACRO(2, 0, ADDR_MOD_7, 0);
        TTI_SFPSHFT2(p_sfpu::LREG0, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPLOADMACRO(3, 0, ADDR_MOD_7, 0);
        TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPLOADMACRO(0, 0, ADDR_MOD_7, 0);
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPLOADMACRO(1, 0, ADDR_MOD_7, 0);
        TTI_SFPSHFT2(p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPLOADMACRO(2, 0, ADDR_MOD_7, 0);
        TTI_SFPSHFT2(p_sfpu::LREG0, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPLOADMACRO(3, 0, ADDR_MOD_7, 0);
        TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPLOADMACRO(0, 0, ADDR_MOD_7, 0);
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPLOADMACRO(1, 0, ADDR_MOD_7, 0);
        TTI_SFPSHFT2(p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPLOADMACRO(2, 0, ADDR_MOD_7, 0);
        TTI_SFPSHFT2(p_sfpu::LREG0, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPLOADMACRO(3, 0, ADDR_MOD_7, 0);
        TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPLOADMACRO(0, 0, ADDR_MOD_7, 0);
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPLOADMACRO(1, 0, ADDR_MOD_7, 0);
        TTI_SFPSHFT2(p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPLOADMACRO(2, 0, ADDR_MOD_7, 0);
        TTI_SFPSHFT2(p_sfpu::LREG0, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPLOADMACRO(3, 0, ADDR_MOD_7, 0);
        TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG14, p_sfpu::LREG4, 5);

        TTI_SFPNOP;
#endif
    } else {
        if constexpr (!is_fp32_dest_acc_en) {
            // _calculate_exponential_tti_bf16_() path:
            // Auto-increment Dest on ADDR_MOD_6
            addr_mod_t{
                .srca = {.incr = 0},
                .srcb = {.incr = 0},
                .dest = {.incr = 2},
            }
                .set(ADDR_MOD_6);

            // LREG12 = 1/ln2
            TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0x3fb8);
            TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0xaa3b);
            TTI_SFPCONFIG(0, p_sfpu::LREG12, 0);

            // LREG13 = c2 = 4.791750143340323e-15f (0x27aca418)
            TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0x27ac);
            TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0xa418);
            TTI_SFPCONFIG(0, p_sfpu::LREG13, 0);
        } else {
            // fp32 scalar path (_sfpu_exp_fp32_accurate_) — uses the scalar
            // reciprocal LLK for negative inputs, so its constants must be
            // primed here.
            sfpu_reciprocal_init<false>();
        }
    }
}

}  // namespace sfpu
}  // namespace ckernel
