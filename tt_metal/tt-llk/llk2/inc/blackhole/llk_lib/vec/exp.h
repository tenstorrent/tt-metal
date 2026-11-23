// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, vec: the exp functions (llk_lib/vec/common.h has the shape of a function file).
//   vec_exp_init / vec_exp_iterations     the exact exp: exp_21f in a 16-bit Dest (Moroz et al.), the fp32 form in a
//                                         32-bit Dest (after N. Juffa), each a body written instruction by instruction and
//                                         replayed from the SFPU's replay slots
//   vec_exp_even_columns_iterations       exp(scale x) on 4 iterations (two rows each: the even columns of a face),
//                                         the exact exp or a polynomial; no init of its own
//   vec_exp_21f, vec_exp_fp32, vec_exp_value, vec_exp_scaled_value   one value through the exp, for the functions
//                                         that are built on it (sigmoid, silu, gelu, softplus)
// The fast exp is a file of its own: llk_lib/vec/fast_exp.h.
// Moved from tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h and
// experimental/llk_sfpu/ckernel_sfpu_sdpa.h (the half-face exp), instruction for instruction.

#pragma once

#include <cstdint>

#include "llk_lib/platform/sfpu.h"
#include "llk_lib/vec/common.h"
#include "llk_lib/vec/recip.h"

#ifdef LLK2_SFPU_THREAD

namespace llk2::llk_lib
{

/*
 * A float to an integer, for exp_21f only. A general conversion branches on special cases (a negative exponent,
 * overflow); in exp_21f they never happen, which allows this branch-free and much smaller form.
 *
 * The constraint on `val` is: 0 <= val < 128.0f
 * Note: the value is taken as already divided by 2^23, so the output is scaled by 2^23 compared to 'val'.
 * Otherwise the shift would be by `exp - 23` instead of `exp`.
 * This saves 1 SFPADDI instruction.
 */
sfpi_inline sfpi::vInt vec_exp_21f_to_int(sfpi::vFloat val)
{
    sfpi::vInt exp = sfpi::exexp(val);
    sfpi::vInt man = sfpi::exman(val, sfpi::MantissaMode::ImplicitOne); // get mantissa with implicit bit (man in [1; 2])
    man            = sfpi::shft(man, exp, sfpi::ShiftMode::Logical);
    return man;
}

// The coefficients of exp_21f's refinement polynomial of 2^x_f, and 1 / ln 2. A function that evaluates the exp
// inside a row loop passes them to vec_exp_21f as sfpi::vFloats built once before the loop (hoisted constants:
// llk_lib/vec/common.h).
constexpr float vec_exp_21f_c0      = 1.0017248f;
constexpr float vec_exp_21f_c1      = 7.839635491371155e-08f;
constexpr float vec_exp_21f_c2      = 4.791750143340323e-15f;
constexpr float vec_exp_21f_one_ln2 = 1.4426950216293334961f; // 0x3FB8AA3B
// c0 as the replayed body loads it: rounded to fp16a (1.001953125), so that one SFPLOADI holds it.
constexpr std::uint16_t vec_exp_21f_c0_fp16a = 0x3c02;

// 127.0f and 255.0f as bf16 immediates: the exponent bias, and the upper clamp of exp_21f's x / ln 2 + 127.
constexpr std::uint16_t vec_bf16_127 = 0x42fe;
constexpr std::uint16_t vec_bf16_255 = 0x437f;

// Programs vConstFloatPrgm1 = 1 / ln 2 for the functions that evaluate an exp per row with hoisted constants (sigmoid,
// silu), which read it on every row. 1 / ln 2 is the same fp32 (0x3FB8AA3B) as the fp32 exp's log2(e), so one register
// serves a function's bf16 and fp32 arms. Prgm0 is left alone: it is the reciprocal's Newton constant
// (vec_recip_newton_init), which those functions need as well.
inline void vec_exp_one_ln2_init()
{
    sfpi::vConstFloatPrgm1 = vec_exp_21f_one_ln2;
}

/*
 * This function implements the exponential function using a polynomial approximation algorithm
 * based on "Simple Multiple Precision Algorithms for Exponential Functions [Tips & Tricks]"
 * by Moroz et al. 2022 (https://doi.org/10.1109/MSP.2022.3157460).
 * More specifically, it is the implementation of the `exp_21f` algorithm described in Section 5
 *
 * This overload takes the four constants from the caller (vec_exp_21f_* above). Each may be a float
 * (materialised by SFPLOADI where it is used, exactly as a literal would be) or an sfpi::vFloat /
 * vConstFloatPrgmN the caller holds across its row loop. The arithmetic is identical either way; only the
 * source of the constants differs. The constants are template-typed rather than plain sfpi::vFloat on
 * purpose: a vFloat parameter is built before the call and stays live through the body, which reorders
 * the SFPLOADIs of the non-hoisting callers (their codegen would no longer be identical) and costs an
 * LReg per constant for the whole body.
 *
 * @param val     The input value (sfpi::vFloat vector), can be any floating point number
 * @param one_ln2 1/ln(2) (vec_exp_21f_one_ln2)
 * @param c0      Constant term of the 2**x_f refinement polynomial (vec_exp_21f_c0)
 * @param c1      Linear coefficient of that polynomial (vec_exp_21f_c1)
 * @param c2      Quadratic coefficient of that polynomial (vec_exp_21f_c2)
 *
 * @return sfpi::vFloat Result of exp(val)
 *
 * @see Moroz et al. 2022 - "Simple Multiple Precision Algorithms for Exponential Functions"
 *      ( https://doi.org/10.1109/MSP.2022.3157460 )
 */
template <bool Fp32, typename K, typename C0, typename C1, typename C2>
sfpi_inline sfpi::vFloat vec_exp_21f(sfpi::vFloat val, K one_ln2, C0 c0, C1 c1, C2 c2)
{
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
    // In our case, we will let the multiplication by 2^23 be done implicitly in vec_exp_21f_to_int
    sfpi::vFloat xlog2 = (val * one_ln2 + 127.f);

    // Intermediary values can overflow in xlog2 is outside of [0, 256[ which leads to invalid results instead of 0
    // (when input < -88.5) and +inf (when input > 88.5)
    // To avoid this, we clamp xlog2 to [0, 255]
    // (thresholds values are rounded to bf16, as it does not change result but only requires one SFPLOADI vs. two)
    xlog2 = sfpi::clamp(xlog2, 0.0f, 255.0f);

    sfpi::vFloat z = sfpi::as<sfpi::vFloat>(vec_exp_21f_to_int(xlog2));

    sfpi::vInt exponential_part = exexp(z, sfpi::ExponentMode::Biased); // Extract exponent ( = 2**(integer part of val/ln2))
    sfpi::vMag fractional_part  = sfpi::exman(z);                       // Extract mantissa ( = leftover part, in [0; 1])

    sfpi::vFloat frac = sfpi::convert<sfpi::vFloat>(fractional_part, sfpi::RoundMode::Nearest);

    // To refine approximation of 2**(x_f), we use an approximation of 2**x on [0; 2^23]
    // This uses a 2nd degree polynomial adjustment of the fractional part
    frac = vec_polynomial::eval(frac, c0, c1, c2);

    // Recombined exponent and mantissa: this is equivalent to 2**(x_i) * 2**(x_f)
    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);

    if constexpr (!Fp32)
    {
        // LRegs work on float32 data. If DST is bfloat16 then SFPSTORE will truncate it.
        // This can reduce accuracy: for instance, 9**2 = 80.8 gets round to 80.5
        // rather than 81 (which would have been correct).
        // To avoid this issue, we explicitly convert to bfloat16 using round-to-nearest.
        y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
    }

    return y;
}

// exp_21f with every constant a literal; see the five-argument overload above.
// @param val The input value (sfpi::vFloat vector), can be any floating point number
// @return sfpi::vFloat Result of exp(val)
template <bool Fp32>
sfpi_inline sfpi::vFloat vec_exp_21f(sfpi::vFloat val)
{
    return vec_exp_21f<Fp32>(val, vec_exp_21f_one_ln2, vec_exp_21f_c0, vec_exp_21f_c1, vec_exp_21f_c2);
}

/*
 * Implementation of vec_exp_21f (same algorithm) with TTI intrinsics
 * This implementation is faster, and give comparable accuracy as vec_exp_21f
 * (~< 1 ULP).
 *
 * Requires vec_exp_init<false>() to have been called to configure
 * vec_addr_mod_step (dest auto-increment by 2 on SFPSTORE) and to load:
 *   - LREG12 = 1/ln2 (sfpi::vConstFloatPrgm0)
 *   - LREG13 = c2    (sfpi::vConstFloatPrgm1)  — poly coeff 4.791750e-15f
 */
template <bool Scaled, bool Fp32, int Iterations>
inline void vec_exp_21f_replayed(const std::uint16_t scale_bf16)
{
    // Iteration-invariant constants. Loaded once before the loop.
    //
    //   LREG5 = 127.0f                      (bias term in z = x/ln2 + 127)
    //   LREG6 = 7.839635491371155e-08f      (poly coeff c1)
    //   LREG7 = 1.001953125f                (poly coeff c0 = 1.0017248f rounded to fp16a, one SFPLOADI)
    //   LREG12 = 1/ln2                      (programmable, set in init)
    //   LREG13 = 4.791750143340323e-15f     (poly coeff c2; programmable, set in init)
    //
    // In-loop scratch:
    //   LREG0 = val → integer-part work
    //   LREG1 = 255 (loaded inside loop) → exexp result → frac (int) → frac (float) → poly result
    //   LREG2 = poly accumulator
    //   LREG3 = xlog2 (preserved through int-part work) → mask
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_FLOATB, vec_bf16_127);

    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_UPPER, vec_hi16(vec_exp_21f_c1));
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_LOWER, vec_lo16(vec_exp_21f_c1));

    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATA, vec_exp_21f_c0_fp16a);

    // Number of instructions in one iteration of the loop body. Used by the
    // replay record and run below; MUST match exactly the
    // count of TTI_ instructions emitted by the recorded lambda, or the replay
    // buffer will misalign.
    //
    //   Base body:                                                15
    //     SFPLOAD, SFPMAD, SFPLOADI(255), SFPSWAP,
    //     SFPEXEXP, SFPEXMAN8, SFPSHFT, SFPEXMAN9, SFPCAST,
    //     SFPMAD poly1, SFPGT mask, SFPMAD poly2,
    //     SFPAND, SFPSETEXP, SFPSTORE.
    //   + Scaled ? 1 : 0                  (SFPMULI scale)
    //   + Fp32 ? 0 : 1       (SFP_STOCH_RND fp32→bf16)
    constexpr int BODY_LEN = 15 + (Scaled ? 1 : 0) + (Fp32 ? 0 : 1);

    // The first iteration records the loop body into replay slots [0, BODY_LEN) while executing it; the other
    // iterations replay it: BODY_LEN + (Iterations - 1) instructions in the binary instead of Iterations * BODY_LEN.
    // The recorded instructions run exactly as if issued inline, and the body's store steps the Dest row counter
    // (vec_addr_mod_step), so each replay is on the next rows.
    //
    // The slots are shared: the other replayed SFPU bodies record into them, and on the math thread eltwise's and
    // reduce's tiles do (math_common.h). Nothing relies on them across calls: every call records again.
    hal::replay::record<hal::replay::BufferRange {0, BODY_LEN}, hal::replay::RecordBehavior::RecordAndExecute>(
        [scale_bf16]
        {
            // val = sfpi::dst_reg[0]
            TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, addr_mod_hold, 0);

            if constexpr (Scaled)
            {
                // Multiply LREG0 by the BF16 scale immediate in-place.
                TTI_SFPMULI(scale_bf16, p_sfpu::LREG0, 0);
            }

            // xlog2 = val * (1/ln2) + 127.0f, into LREG3 (preserved past int-part work).
            TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG12, p_sfpu::LREG5, p_sfpu::LREG3, 0);

            // LREG1 = 255.0f. Slots into the SFPMAD's 2-cycle latency window.
            TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, vec_bf16_255);

            // Upper clamp. SFPSWAP (mode VEC_MIN_MAX = "max into lreg_dest"):
            //   LREG1 = max(255, xlog2), LREG3 = min(255, xlog2).
            TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);

            // vec_exp_21f_to_int: shift mantissa left by exp-bias bits.
            // Reads xlog2 from LREG3, leaves int_part in LREG0 (LREG0 freed of val).
            TTI_SFPEXEXP(0, p_sfpu::LREG3, p_sfpu::LREG1, 0); // LREG1 = exexp(xlog2)
            TTI_SFPEXMAN(0, p_sfpu::LREG3, p_sfpu::LREG0, 0); // LREG0 = exman8(xlog2)
            TTI_SFPSHFT(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);  // LREG0 <<= LREG1   (int_part)

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

            if constexpr (!Fp32)
            {
                // Round float32 -> bfloat16 using round-to-nearest before
                // SFPSTORE truncates. Avoids ULP loss on values like 9*9 = 80.8.
                TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0, p_sfpu::LCONST_0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
            }

            // sfpi::dst_reg[0] = y; sfpi::dst_reg++;
            // (vec_addr_mod_step increments dest by 2 on store.)
            TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, vec_addr_mod_step, 0);
        });

#pragma GCC unroll 8
    for (std::uint32_t i = 1; i < Iterations; i++)
    {
        // Replay the recorded body for the remaining Iterations - 1 elements.
        // Each replay is one REPLAY-equivalent issue; the body executes as
        // if it had been issued inline, dst_reg advancing via vec_addr_mod_step.
        hal::replay::run<hal::replay::BufferRange {0, BODY_LEN}>();
    }
}

// Utility function to round a float to a 32-bit integer while also calculating the
// integer part of the rounded value
sfpi_inline sfpi::vFloat vec_round_to_nearest_int(sfpi::vFloat z, sfpi::vInt& k_int)
{
    // From Hacker's Delight: round-to-nearest method
    // float -> int32 (round to nearest even): n = (x + float(c231)) - int32(c231)
    // round-to-nearest: n = (x + float(c231)) - float(c231)
    // where c231 = 0x4B400000 (2^23 + 2^22)
    const sfpi::vFloat c231 = vec_float_of(0x4B400000U); // 2^23 + 2^22

    sfpi::vFloat tmp = z + c231;
    sfpi::vFloat k   = tmp - c231;
    k_int            = sfpi::as<sfpi::vInt>(tmp) - sfpi::as<sfpi::vInt>(c231);

    return k;
}

/*
 * The vec_exp_fp32 code is derived from code by Norbert Juffa.
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

// Juffa exp constants that the hoisted-constant overload below takes from its caller (as vec_exp_21f takes
// vec_exp_21f_*). log2(e) is the same fp32 as vec_exp_21f_one_ln2, so one Prgm register can serve both exps.
constexpr float vec_exp_fp32_log2e      = vec_exp_21f_one_ln2; // log2(e) == 1/ln(2), 0x3FB8AA3B
constexpr float vec_exp_fp32_neg_ln2_hi = -6.93145752e-1f;
constexpr float vec_exp_fp32_p0         = 1.37805939e-3f;
constexpr float vec_exp_fp32_p1         = 8.37312452e-3f; // 0x1.125edcp-7

// Non-finite behaviour of the guarded base-e form (unsafe = false, base_two = false), measured on
// Blackhole silicon:
//   +-NaN (either sign, quiet or signalling) -> NaN    +Inf -> +Inf    -Inf -> +0
// unsafe = true drops both guards and preserves none of this.
//
// base_two = true computes 2^a instead of e^a. The reduction is then exact: j = round(a) and
// a - j are both representable, so the only rounding before the polynomial is one multiply by
// ln 2 on |a - j| <= 1/2. Passing a * ln2 to the base-e form instead rounds the argument,
// and that relative error grows into the result in proportion to |a| (tens of ULP near 2^-126).
//
// This overload takes log2(e), -ln2_hi and the two leading polynomial coefficients from the caller, each
// either a float (materialised where used, as a literal is) or a vFloat / vConstFloatPrgmN held across a
// row loop. They are template-typed rather than plain sfpi::vFloat for the reason given at
// vec_exp_21f, and here it matters more: vFloat parameters stay live through the body, which can push a caller
// that has other values live around this exp past the eight LRegs ("too few lregs"; it did in 1.0's tanhshrink).
// The base_two form reads neither log2(e) nor -ln2_hi.
// vec_exp_fp32(a) below passes the literals and is what everything not hoisting reads.
template <bool unsafe = false, bool base_two = false, typename K, typename H, typename P0, typename P1>
sfpi_inline sfpi::vFloat vec_exp_fp32(sfpi::vFloat a, K log2e, H neg_ln2_hi, P0 p0, P1 p1)
{
    sfpi::vInt i, e;
    sfpi::vFloat f, r, j, y;
    sfpi::vSMag16 sm;

    if constexpr (base_two)
    {
        // j = round(a), f = (a - j) * ln2
        r  = p0;
        sm = sfpi::convert<sfpi::vSMag16>(a, sfpi::RoundMode::Nearest);
        j  = sfpi::convert<sfpi::vFloat>(sm, sfpi::RoundMode::Nearest);
        f  = (a - j) * 6.93147182e-1f;
    }
    else
    {
        // j = round(a / ln2)
        // interleaved with first coefficient of polynomial
        j  = log2e * a;
        r  = p0;
        sm = sfpi::convert<sfpi::vSMag16>(j, sfpi::RoundMode::Nearest);
        j  = sfpi::convert<sfpi::vFloat>(sm, sfpi::RoundMode::Nearest);

        // f = a - i*j (two-part cody-waite)
        f = j * neg_ln2_hi + a;
        f = j * -1.42860677e-6f + f;
    }

    // approximate r = exp(f) on [-ln2/2, ln2/2]
    // interleaved with conversion of i from sign-mag to two's complement via abs and copysgn
    r = r * f + p1;
    r = r * f + 4.16695364e-2f; // 0x1.555b5ap-5
    r = r * f + 1.66664720e-1f; // 0x1.555450p-3
    r = r * f + 4.99999851e-1f; // 0x1.fffff6p-2
    i = sfpi::abs(sfpi::as<sfpi::vInt>(sm));
    y = r * f + 1.0f;
    i = sfpi::as<sfpi::vInt>(sfpi::copysgn(sfpi::as<sfpi::vFloat>(i), j));
    r = y * f + 1.0f;

    if constexpr (unsafe)
    {
        // y = 2**i * r
        e = sfpi::exexp(r, sfpi::ExponentMode::Biased) + i;
        y = sfpi::setexp(r, e);
    }
    else
    {
        // overflow: y = infinity or NaN
        y *= std::numeric_limits<float>::infinity();

        e = sfpi::exexp(r, sfpi::ExponentMode::Biased) + i;
        v_if (sfpi::nearby(e < 255))
        {
            // y = 2**i * r
            y = sfpi::setexp(r, e);

            v_if (sfpi::nearby(e < 1))
            {
                // underflow, including subnormals
                y = 0.0f;
            }
            v_endif;
        }
        v_endif;
    }

    return y;
}

template <bool unsafe = false, bool base_two = false>
sfpi_inline sfpi::vFloat vec_exp_fp32(sfpi::vFloat a)
{
    return vec_exp_fp32<unsafe, base_two>(a, vec_exp_fp32_log2e, vec_exp_fp32_neg_ln2_hi, vec_exp_fp32_p0, vec_exp_fp32_p1);
}

// The constants of vec_exp_fp32<false> as vec_exp_fp32_replayed holds them. 1/ln2, the high
// part of -ln2 and the first two polynomial coefficients are the hoisted vec_exp_fp32_log2e, vec_exp_fp32_neg_ln2_hi,
// vec_exp_fp32_p0 and vec_exp_fp32_p1 above (P0 as the fp16 immediate below); the rest are only here. LREG12..14
// (LOG2E, NEG_LN2_HI, NEG_LN2_LO) are programmed by vec_exp_init<true>, LREG4..7 (P1, C2..C4)
// are loaded per call, r0 stays an fp16 immediate.
constexpr float vec_exp_fp32_neg_ln2_lo       = -1.42860677e-6f; // LREG14, 0xB5BFBE8E
constexpr float vec_exp_fp32_c2               = 4.16695364e-2f;  // LREG5, 0x3D2AADAD
constexpr float vec_exp_fp32_c3               = 1.66664720e-1f;  // LREG6, 0x3E2AAA28
constexpr float vec_exp_fp32_c4               = 4.99999851e-1f;  // LREG7, 0x3EFFFFFB
constexpr std::uint16_t vec_exp_fp32_r0_fp16a = 0x15A5;          // vec_exp_fp32_p0 = 1.37805939e-3f, exact in fp16

constexpr std::uint16_t vec_exp_fp32_inf_fp16b = 0x7F80; // +inf as an fp16b SFPMULI immediate

/**
 * @brief vec_exp_fp32<false> over Iterations vectors, its head replayed from the SFPU's replay slots.
 *
 * Same instructions, operand roles and order as the sfpi form, so the result is bit-identical to it; the constants
 * come from LREG4..7 and LREG12..14 instead of per-vector SFPLOADIs, the sign-magnitude to two's complement step
 * is one SFPCAST, and the store advances DEST by vec_addr_mod_step.
 *
 * The body is 24 instructions (25 with Scaled). Blackhole splits each thread's 32-entry replay buffer: slots
 * [0, matmul_replay_start) are free for the SFPU and the rest hold matmul's tile (matmul_configure records its 16
 * MVMULs there and matmul_execute replays them). On the math thread eltwise and reduce record their tiles in
 * [0, matmul_replay_start) too, and record them again in their next configure (math_common.h). So only the
 * head of the body, the input load through y *= inf, is recorded into slots [0, HEAD_LEN) and replayed; the
 * 9-instruction tail (exponent assembly, the overflow and underflow lanes, the store) is issued inline per vector.
 *
 * @tparam Scaled: Multiply the input by scale_bf16 first
 * @tparam Iterations: Vectors per call
 * @param scale_bf16: fp16b scale, read when Scaled.
 * @note Requires vec_exp_init<true> (LREG12..14 and vec_addr_mod_step). Writes LREG0..7 and replay
 * slots [0, 15 + Scaled).
 * @note LREG12..14 are the SFPU's shared programmable constants (sfpi::vConstFloatPrgm0..2). Other functions'
 * inits (reciprocal, tanh, selu, softplus, ...) reprogram them. In LLK 2.0 that cannot come between: vec_configure
 * runs vec_exp_init directly in front of the iterations of every call. The bf16 body (vec_exp_21f_replayed) takes
 * LREG12..13 from vec_exp_init the same way.
 */
template <bool Scaled, int Iterations>
inline void vec_exp_fp32_replayed(const std::uint32_t scale_bf16)
{
    platform::sfpu::load_immediate32<platform::sfpu::Lreg::L4, vec_bits_of(vec_exp_fp32_p1)>();
    platform::sfpu::load_immediate32<platform::sfpu::Lreg::L5, vec_bits_of(vec_exp_fp32_c2)>();
    platform::sfpu::load_immediate32<platform::sfpu::Lreg::L6, vec_bits_of(vec_exp_fp32_c3)>();
    platform::sfpu::load_immediate32<platform::sfpu::Lreg::L7, vec_bits_of(vec_exp_fp32_c4)>();

    // Instructions the head emits; MUST match the lambda exactly (the record does not check the count, and a
    // mismatch misaligns every replay):
    //   SFPLOAD, [SFPMULI scale], SFPMUL, SFPLOADI r0, SFP_STOCH_RND, SFPCAST,
    //   2 SFPMAD (f), 4 SFPMAD (Horner), SFPCAST, 2 SFPMAD (y, r), SFPMULI inf      = 15 + Scaled
    constexpr std::uint32_t HEAD_LEN = 15 + (Scaled ? 1 : 0);
    static_assert(HEAD_LEN <= matmul_replay_start, "the head must stay below matmul's replay slots");

    auto head = [scale_bf16]
    {
        // LREG1 = a; LREG3 = j = a / ln2; LREG0 = r0
        TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, addr_mod_hold, 0);
        if constexpr (Scaled)
        {
            TT_SFPMULI(scale_bf16, p_sfpu::LREG1, 0);
        }
        TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG12, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATA, vec_exp_fp32_r0_fp16a);
        // LREG3 = sm = round(j) as a sign-magnitude int16; LREG2 = j = float(sm)
        TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_NEAREST, 0, p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPSTOCHRND_MOD1_FP32_TO_SMAG16);
        TTI_SFPCAST(p_sfpu::LREG3, p_sfpu::LREG2, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
        // LREG1 = f = a - j ln2, two parts
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG13, p_sfpu::LREG1, p_sfpu::LREG1, 0);
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG14, p_sfpu::LREG1, p_sfpu::LREG1, 0);
        // LREG0 = r = (((r0 f + c1) f + c2) f + c3) f + c4;
        // LREG3 = i = two's complement of sm, in the chain's shadow
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG0, 0);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG5, p_sfpu::LREG0, 0);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG6, p_sfpu::LREG0, 0);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG7, p_sfpu::LREG0, 0);
        // INT32_TO_SM32 is its own inverse on sign-magnitude values, so on sm it gives the two's complement the
        // sfpi form builds with abs and copysgn. Exact because FP32_TO_SMAG16 never produces SM -0, the one input
        // where the two would differ.
        TTI_SFPCAST(p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPCAST_MOD1_INT32_TO_SM32);
        // LREG0 = y = r f + 1; LREG1 = r = y f + 1; y *= inf
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG0, 0);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG1, 0);
        TTI_SFPMULI(vec_exp_fp32_inf_fp16b, p_sfpu::LREG0, 0);
    };
    auto tail = []
    {
        // LREG2 = e = biased exponent of r + i; lanes with e < 255: y = setexp(r, e); lanes with e < 1: y = 0
        TTI_SFPEXEXP(0, p_sfpu::LREG1, p_sfpu::LREG2, sfpi::SFPEXEXP_MOD1_NODEBIAS);
        TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPIADD(0xF01 /* -255 */, p_sfpu::LREG2, p_sfpu::LREG3, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_LT0);
        TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
        TTI_SFPSETEXP(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        TTI_SFPIADD(0xFFF /* -1 */, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_LT0);
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, vec_addr_mod_step, 0);
    };

    // The first vector records the head while executing it; hal::replay::record brackets the record with
    // disable_gathering()/enable_gathering() when ENABLE_GATHERING is defined.
    hal::replay::record<hal::replay::BufferRange {0, HEAD_LEN}, hal::replay::RecordBehavior::RecordAndExecute>(head);
    tail();
#pragma GCC unroll 8
    for (int d = 1; d < Iterations; d++)
    {
        hal::replay::run<hal::replay::BufferRange {0, HEAD_LEN}>();
        tail();
    }
}

/// exp(val), one value: exp_21f in a 16-bit Dest (rounded to bf16), the fp32 form in a 32-bit Dest.
template <bool Fp32>
sfpi_inline sfpi::vFloat vec_exp_value(sfpi::vFloat val)
{
    if constexpr (Fp32)
    {
        return vec_exp_fp32(val);
    }
    else
    {
        return vec_exp_21f<false>(val);
    }
}

/// exp(scale val), one value; the scale is a bf16 pattern, read when Scaled.
template <bool Scaled, bool Fp32>
sfpi_inline sfpi::vFloat vec_exp_scaled_value(sfpi::vFloat val, const std::uint32_t scale_bf16)
{
    if constexpr (Scaled)
    {
        val = val * sfpi::sFloat16b(scale_bf16);
    }
    sfpi::vFloat result = vec_exp_value<Fp32>(val);
    return result;
}

/// The exact exp on Iterations iterations from the Dest row counter on: the replayed exp_21f body in a 16-bit Dest,
/// the replayed fp32 body in a 32-bit Dest. With DISABLE_SFPLOADMACRO (the ttsim build) the fp32 form is the SFPI C++
/// one: the plain reference stream. The scale is a bf16 pattern, read when Scaled. Needs vec_exp_init.
template <bool Fp32, bool Scaled, int Iterations>
void vec_exp_iterations(const std::uint32_t scale_bf16)
{
    if constexpr (!Fp32)
    {
        vec_exp_21f_replayed<Scaled, Fp32, Iterations>(scale_bf16);
    }
    else
    {
        if constexpr (!vec_has_loadmacro)
        {
            for (int d = 0; d < Iterations; d++)
            {
                sfpi::vFloat val = sfpi::dst_reg[0];
                sfpi::dst_reg[0] = vec_exp_scaled_value<Scaled, Fp32>(val, scale_bf16);
                sfpi::dst_reg++;
            }
        }
        else
        {
            vec_exp_fp32_replayed<Scaled, Iterations>(scale_bf16);
        }
    }
}

/// The exact exp's part of vec_configure: the stepping address modifier its last store names and its programmable
/// constants:
///   16-bit Dest  L12 = 1 / ln 2, L13 = c2 of exp_21f's polynomial
///   32-bit Dest  L12 = log2(e), L13, L14 = -ln 2 in two parts; with DISABLE_SFPLOADMACRO (the SFPI C++ form) only
///                the reciprocal's Newton constant
template <bool Fp32>
void vec_exp_init()
{
    using namespace platform::sfpu;
    if constexpr (!Fp32)
    {
        set_addr_mod<vec_addr_mod_step, counter_stay, counter_stay, counter_step {.incr = vec_rows_per_iteration_step}>();

        // L12 = 1 / ln 2 and L13 = c2. The upper half first: vec_set_constant loads the lower half first.
        load_immediate<LoadImmediate {.destination = Lreg::L0, .half = ImmediateHalf::Upper, .value = vec_hi16(vec_exp_21f_one_ln2)}>();
        load_immediate<LoadImmediate {.destination = Lreg::L0, .half = ImmediateHalf::Lower, .value = vec_lo16(vec_exp_21f_one_ln2)}>();
        configure<Configure {.target = ConfigTarget::Constant12}>();

        load_immediate<LoadImmediate {.destination = Lreg::L0, .half = ImmediateHalf::Upper, .value = vec_hi16(vec_exp_21f_c2)}>();
        load_immediate<LoadImmediate {.destination = Lreg::L0, .half = ImmediateHalf::Lower, .value = vec_lo16(vec_exp_21f_c2)}>();
        configure<Configure {.target = ConfigTarget::Constant13}>();
    }
    else
    {
        if constexpr (!vec_has_loadmacro)
        {
            vec_recip_newton_init<false>();
        }
        else
        {
            set_addr_mod<vec_addr_mod_step, counter_stay, counter_stay, counter_step {.incr = vec_rows_per_iteration_step}>();
            vec_set_constant<ConfigTarget::Constant12, vec_bits_of(vec_exp_fp32_log2e)>();
            vec_set_constant<ConfigTarget::Constant13, vec_bits_of(vec_exp_fp32_neg_ln2_hi)>();
            vec_set_constant<ConfigTarget::Constant14, vec_bits_of(vec_exp_fp32_neg_ln2_lo)>();
        }
    }
}

/// exp(ScaleBf16 x) by range reduction and a polynomial, instruction by instruction, on Iterations iterations of two
/// rows each (the Dest row counter moves on by 4 per iteration). Degree is 2 or 4. What each register holds:
///   L2      x, then x * scale
///   L0      x * scale / ln 2; then r = x * scale - k ln 2; then p(r)
///   L1      k = round(x * scale / ln 2), as an int8 and then as a float; then k + 127
///   L2      the Horner sums of p(r) = c0 + r (c1 + r (c2 + ...)); then 2^k (the float with exponent k + 127 and
///           mantissa 0); then the result p(r) * 2^k, which is 0 in the lanes where k + 127 is 0
///   L7, L6, L5 (degree 4: and L4, L3)     c0, c1, c2 (c3, c4)
///   L3, L4  degree 2 only: 1 / ln 2 and -ln 2, for the whole loop. With degree 4 the coefficients take the two
///           registers, and each iteration loads the two constants again (into L1 and L0)
template <int Iterations, int Degree, bool Fp32, std::uint16_t ScaleBf16>
inline void vec_exp_polynomial_iterations()
{
    static_assert(Degree == 2 || Degree == 4, "the polynomial exp runs degree 2 (16-bit Dest) or 4 (32-bit Dest)");
    constexpr bool ln2_in_registers = Degree == 2;

    constexpr float LN2_RECIP = 1.44269504088896340736f;
    constexpr float M_LN2     = -0.69314718055994530942f;

    constexpr float c0 = (Degree == 2) ? 0.999848792924395313327307061545061386175496934006f : 1.0000001510806179002040134468008959160576106495165f;
    constexpr float c1 = (Degree == 2) ? 1.01508760098521056684783640695492761469306929535975f : 0.99996228117047652035114096488703457970402030983204f;
    constexpr float c2 = (Degree == 2) ? 0.50628367056745568861842335616023694454759126020461f : 0.49998365704615426417337683145647067790385638465486f;
    constexpr float c3 = 0.16792157982882225102649214918047336097544632172075f;
    constexpr float c4 = 4.1959439860014343843000081999668024587178974865521e-2;

    if constexpr (Degree == 4)
    {
        TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, vec_lo16(c4));
        TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, vec_hi16(c4));
        TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, vec_lo16(c3));
        TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, vec_hi16(c3));
    }
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_LOWER, vec_lo16(c2));
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_UPPER, vec_hi16(c2));
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_LOWER, vec_lo16(c1));
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_UPPER, vec_hi16(c1));
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, vec_lo16(c0));
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, vec_hi16(c0));

    if constexpr (ln2_in_registers)
    {
        TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, vec_lo16(LN2_RECIP));
        TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, vec_hi16(LN2_RECIP));
        TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, vec_lo16(M_LN2));
        TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, vec_hi16(M_LN2));
    }

    for (int d = 0; d < Iterations; d++)
    {
        constexpr InstrModLoadStore input_type = Fp32 ? InstrModLoadStore::FP32 : InstrModLoadStore::FP16B;
        TTI_SFPLOAD(p_sfpu::LREG2, input_type, addr_mod_hold, 0);

        // x * scale
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, ScaleBf16);
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);

        // k = round(x * scale / ln 2)
        if constexpr (ln2_in_registers)
        {
            TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        }
        else
        {
            TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_LOWER, vec_lo16(LN2_RECIP));
            TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_UPPER, vec_hi16(LN2_RECIP));
            TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        }
        TTI_SFP_STOCH_RND(0, 0, p_sfpu::LCONST_0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPSTOCHRND_MOD1_FP32_TO_INT8);
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);

        // r = x * scale - k ln 2
        if constexpr (ln2_in_registers)
        {
            TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG2, p_sfpu::LREG0, 0);
        }
        else
        {
            TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, vec_lo16(M_LN2));
            TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, vec_hi16(M_LN2));
            TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
        }

        // p(r), Horner from the top coefficient
        if constexpr (Degree == 2)
        {
            TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LREG6, p_sfpu::LREG2, 0);
            TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG7, p_sfpu::LREG0, 0);
        }
        else
        {
            TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG4, p_sfpu::LREG2, 0);
            TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG5, p_sfpu::LREG2, 0);
            TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG6, p_sfpu::LREG2, 0);
            TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG7, p_sfpu::LREG0, 0);
        }

        // 2^k: k + 127 rounded to a uint8, as the exponent of a float with mantissa 0; then p(r) * 2^k
        TT_SFPADDI(vec_bf16_127, p_sfpu::LREG1, 0);
        TTI_SFP_STOCH_RND(0, 0, p_sfpu::LCONST_0, p_sfpu::LREG1, p_sfpu::LREG2, sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT8);
        TTI_SFPSETEXP(0, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);

        // The lanes where k + 127 is 0: the result is 0
        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, 0);
        TTI_SFPENCC(0, 0, 0, 0);

        if constexpr (!Fp32)
        {
            TTI_SFP_STOCH_RND(0, 0, p_sfpu::LCONST_0, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        }
        TTI_SFPSTORE(p_sfpu::LREG2, input_type, addr_mod_hold, 0);
        TTI_INCRWC(0 /*rwc_cr*/, 4 /*rwc_d*/, 0 /*rwc_b*/, 0 /*rwc_a*/);
    }
}

/// exp's half-face body (VecRegion::LeftFacesEvenColumns): exp(ScaleBf16 x) on 4 iterations of two rows each, so on
/// the even columns of one face. Exact: the exact exp (vec_exp_scaled_value); otherwise the polynomial form
/// (VecForm::Polynomial), degree 2 (16-bit Dest) or 4.
template <bool Exact, std::uint16_t ScaleBf16, bool Fp32>
inline void vec_exp_even_columns_iterations()
{
    constexpr int ITERATIONS_HALF_FACE = 4;
    if constexpr (Exact)
    {
        for (int d = 0; d < ITERATIONS_HALF_FACE; d++)
        {
            sfpi::vFloat val    = sfpi::dst_reg[0];
            sfpi::vFloat result = vec_exp_scaled_value<true /*Scaled*/, Fp32>(val, ScaleBf16);
            sfpi::dst_reg[0]    = result;
            sfpi::dst_reg += 2;
        }
    }
    else
    {
        constexpr int polynomial_degree = Fp32 ? 4 : 2;
        vec_exp_polynomial_iterations<ITERATIONS_HALF_FACE, polynomial_degree, Fp32, ScaleBf16>();
    }
}

} // namespace llk2::llk_lib

#endif // LLK2_SFPU_THREAD
