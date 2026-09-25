// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_polyval.h"
#include "ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_load_config.h"
#include "cmath_common.h"

namespace ckernel {
namespace sfpu {

// Legacy tanh derivative: 1 - lut(x)^2, with tanh taken from the SFPLUT rather than computed.
// For finite |x| >= 3 the result is exactly 0, so every point in the tail is 100% relative
// error against a sech^2 that is merely small -- and, measured in ulp of that bfloat16
// reference, a flat ~256 rather than anything unbounded. Absolute error is the only metric
// that stays meaningful there, and it stays below sech^2(3) = 0.0099; max absolute error is
// 0.0143 overall (Wormhole, fp32 end to end, every finite bfloat16 input). The infinities are
// the exception to "exactly 0": the tail pair is (A=0, B=1) and the hardware evaluates
// A*|x| + B, so 0 * inf + 1 is NaN and 1 - NaN^2 is NaN. Kept for backward
// compatibility -- calculate_tanh_derivative_sech2 is correctly rounded in bfloat16 instead.
// Nothing in this repository calls it: tanh_derivative_tile dispatches
// calculate_tanh_derivative_sech2 unconditionally and ignores fast_and_approx, and the
// LLK harness runs tt-llk's _calculate_tanh_derivative_ rather than this copy. The table
// in tanh_derivative_init below is live even though this function is not -- see there.
template <bool APPROXIMATION_MODE, int WITH_PRECOMPUTED_TANH = 0, int ITERATIONS = 8>
inline void calculate_tanh_derivative() {
    sfpi::vLut16ss s01 = l_reg[sfpi::LRegs::LReg0];
    sfpi::vLut16ss s23 = l_reg[sfpi::LRegs::LReg1];
    sfpi::vLut16ss s45 = l_reg[sfpi::LRegs::LReg2];
    sfpi::vLut16ii i01 = l_reg[sfpi::LRegs::LReg4];
    sfpi::vLut16ii i23 = l_reg[sfpi::LRegs::LReg5];
    sfpi::vLut16ii i45 = l_reg[sfpi::LRegs::LReg6];

    // tanh'(x) = 1 - (tanh(x))^2
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];

        if constexpr (!WITH_PRECOMPUTED_TANH) {
            val = sfpi::lut(val, s01, i01, s23, i23, s45, i45, sfpi::LutSign::Retain);
        }

        val = val * (-val) + 1.0f;
        sfpi::dst_reg[0] = val;

        sfpi::dst_reg++;
    }

    sfpi::l_reg[LRegs::LReg0] = s01;
    sfpi::l_reg[LRegs::LReg1] = s23;
    sfpi::l_reg[LRegs::LReg2] = s45;
    sfpi::l_reg[LRegs::LReg4] = i01;
    sfpi::l_reg[LRegs::LReg5] = i23;
    sfpi::l_reg[LRegs::LReg6] = i45;
}

template <bool APPROXIMATION_MODE>
inline void tanh_derivative_init() {
    // A 6-entry SFPLUTFP32 FP16 table, TABLE1 breakpoints |x| = 0.5, 1, 1.5, 2, 3, evaluated
    // as 1 - lut(x)^2. Its consumer is not calculate_tanh_derivative above but tt-llk's
    // _calculate_tanh_derivative_, paired with this init under SfpuType::tanh_derivative_lut;
    // the table crosses repositories in LReg0/1/2 (slopes) and LReg4/5/6 (intercepts) and is
    // the whole of that kernel's approximation. Fitted for sech^2, not tanh: tanh_init's own
    // table measures 0.0179 here and is not monotone once squared, against 0.0143 for this.
    //
    // Four properties to preserve if you retune. Segment 0's intercept stays 0, so tanh'(0)
    // is exactly 1. The last segment stays (0, 1.0), which makes the result exactly 0 for
    // finite |x| past 3 and is what bounds it at all -- any nonzero slope there sends
    // 1 - lut^2 to -inf. No segment may reach lut > 1, or the result goes negative. And the
    // lut must not step down at a breakpoint, which is what keeps the result monotone in |x|.
    //
    // That tail entry saturates the finite range only. The hardware evaluates A*|x| + B, so
    // an infinite input computes 0 * inf + 1 = NaN and the kernel returns NaN rather than 0;
    // do not read (0, 1.0) as handling the infinities.
    //
    // UnarySFPUGolden._tanh_derivative_lut mirrors these six pairs by hand, and
    // test_tanh_lut_consistency.py holds all three copies together.
    sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vLut16ss(0.93701171875f, 0.5869140625f);
    sfpi::l_reg[sfpi::LRegs::LReg4] = sfpi::vLut16ii(0.0f, 0.183837890625f);

    sfpi::l_reg[sfpi::LRegs::LReg1] = sfpi::vLut16ss(0.277099609375f, 0.11181640625f);
    sfpi::l_reg[sfpi::LRegs::LReg5] = sfpi::vLut16ii(0.49365234375f, 0.74169921875f);

    sfpi::l_reg[sfpi::LRegs::LReg2] = sfpi::vLut16ss(0.03070068359375f, 0.0f);
    sfpi::l_reg[sfpi::LRegs::LReg6] = sfpi::vLut16ii(0.90625f, 1.0f);
}

// Cody-Waite constants for the inline exp below. -ln(2) is split into a high part that is
// a 13-bit multiple of 2^-13, so k*SECH2_LN2_HI is exact for the |k| it reaches, and a low
// part that carries the rest.
constexpr float SECH2_INV_LN2 = 1.4426950408889634f;
constexpr float SECH2_LN2_HI = -0.6931152343750000f;  // -ln(2) high bits (exact in float)
constexpr float SECH2_LN2_LO = -3.19461832987e-05f;   // -ln(2) low bits

// Cody-Waite range reduction t = k·ln(2) + r, k = round(t/ln2), |r| <= ln(2)/2.
// Returns r and leaves k in k_int for the 2^k scaling.
sfpi_inline sfpi::vFloat sech2_reduce_ln2(sfpi::vFloat t, sfpi::vInt& k_int) {
    sfpi::vFloat z = t * SECH2_INV_LN2;
    sfpi::vFloat k = _sfpu_round_to_nearest_int32_(z, k_int);

    sfpi::vFloat r = k * SECH2_LN2_HI + t;  // Extended precision subtraction
    return k * SECH2_LN2_LO + r;
}

// =============================================================================
// Inline exp for sech²: computes 4*exp(-2|x|)
// =============================================================================
// Cody-Waite range reduction, a polynomial for exp(r), and 2^k applied directly to the
// exponent field, with an explicit FTZ test. Two choices matter at fp32 precision:
//
// 1. The argument is -2|x| exactly (doubling never rounds) and the x4 is applied to
//    the reconstructed exponent rather than folded into the argument as +ln4.
//    Folding it in costs up to ulp(88)/2 = 3.8e-6 of absolute argument error at the
//    far end, which is 3.8e-6 of relative error in the result -- under a bfloat16
//    ulp, but ~60 fp32 ulp. Folding the x4 into the exponent before the FTZ test keeps
//    what the ln4 trick bought: 4*exp(-2|x|) is still a normal fp32 number out to
//    |x| ~ 44.4, well past the |x| = 43.7 where exp(-2|x|) alone goes subnormal and
//    flushes.
// 2. The polynomial is the fp32 degree-6 minimax that _sfpu_exp_fp32_accurate_ in
//    ckernel_sfpu_exp.h uses on the same reduced range. A degree-4 Taylor, which is
//    enough for bfloat16, truncates at r^5/120 = 4.2e-5 relative for |r| <= ln(2)/2.
//
// k*SECH2_LN2_HI is exact (a 13-bit multiple of 2^-13 times |k| <= 130, the bound the
// caller's a < TAIL_REGION_LIMIT guard gives), and so is its sum with t, so the reduced
// argument carries only the SECH2_LN2_LO rounding.
// =============================================================================
sfpi_inline sfpi::vFloat inline_exp4_neg2x_fp32(sfpi::vFloat a) {
    // Degree-6 minimax for exp(r), |r| <= ln(2)/2, from _sfpu_exp_fp32_accurate_
    constexpr float C2 = 4.99999851e-1f;  // 0x1.fffff6p-2
    constexpr float C3 = 1.66664720e-1f;  // 0x1.555450p-3
    constexpr float C4 = 4.16695364e-2f;  // 0x1.555b5ap-5
    constexpr float C5 = 8.37312452e-3f;  // 0x1.125edcp-7
    constexpr float C6 = 1.37805939e-3f;

    sfpi::vFloat t = a * -2.0f;

    // Cody-Waite range reduction: t = k*ln(2) + r
    sfpi::vInt k_int;
    sfpi::vFloat r = sech2_reduce_ln2(t, k_int);

    sfpi::vFloat poly = PolynomialEvaluator::eval(r, 1.0f, 1.0f, C2, C3, C4, C5, C6);

    // 2^(k+2) scaling via direct exponent bit manipulation. The +2 is the x4.
    sfpi::vInt p_exp = sfpi::exexp(poly, sfpi::ExponentMode::Biased);
    sfpi::vInt new_exp = p_exp + k_int + 2;

    // FTZ: if the scaled exponent underflows, result is 0
    sfpi::vFloat result = 0.0f;
    v_if(new_exp > 0) { result = sfpi::setexp(poly, new_exp); }
    v_endif;

    return result;
}

// Newton reciprocal for a strictly positive, well-scaled argument. The only caller
// passes (1 + exp(-2|x|))^2, which lives in [1, 4], so none of the zero / infinity /
// NaN guards in sfpu_reciprocal_iter are reachable. Both bounds are attained and the
// interval is closed, not half-open: x = 0 gives exactly 4, and the exp flushes to
// zero for |x| past ~44.4, which gives exactly 1. Two iterations from the SFPARECIP
// seed, matching the fp32 arm of _sfpu_reciprocal_gt0_ in ckernel_sfpu_trigonometry.h.
// This is also why tanh_derivative_sech2_init needs no setup: the seed is SFPARECIP and
// every constant is a literal. sfpu_reciprocal_iter reads vConstFloatPrgm0 instead, so
// swapping it in here would need an init that loads it.
sfpi_inline sfpi::vFloat inline_reciprocal_1_to_4(sfpi::vFloat x) {
    sfpi::vFloat y = sfpi::approx_recip(x);
    sfpi::vFloat e = -x * y + 1.0f;
    y = y * e + y;
    e = -x * y + 1.0f;
    y = y * e + y;
    return y;
}

// =============================================================================
// Accurate tanh derivative
// =============================================================================
// The exact identity, not an approximation of the shape:
//
//   sech²(x) = 4e / (1 + e)²,  e = exp(-2|x|)
//
// It avoids the catastrophic cancellation in 1 - tanh²(x) (Max ULP = 15,140) and makes no
// external function call (_sfpu_exp_f32_accurate_, sfpu_reciprocal_iter). One formula
// over the whole range and one for both destination formats; a bfloat16 destination only
// adds the final round-to-nearest-even.
//
// This replaces a two-piece fit (degree-10 polynomial below |x| = 3, a degree-4 exp
// tail above it) that was used for both destinations. Each piece was accurate to 1 bf16
// ULP and no further -- 8.0e-4 relative at x = 0, 5.0e-3 just past x = 3, with a step
// *up* across the boundary -- which is tens of thousands of fp32 ULP, and still misrounded
// 292 bfloat16 results. See tenstorrent/tt-metal#57509.
//
// Accuracy, Blackhole, exhaustive against mpmath (negative inputs are the same
// computation, see the sign clear below):
//   fp32 destination: max 2 ULP, at 742 of the 2.14e9 non-negative inputs, every other
//     input within 1 ULP; 99.0% correctly rounded (the two-piece fit: 65,345 max).
//   bf16 destination, and bf16 data through an fp32 destination: every one of the
//     65,280 finite inputs correctly rounded (the two-piece fit: 292 at 1 ULP).
// Performance, MATH_ISOLATE, ITERATIONS=32, cycles/tile: 2683.5 with an fp32
// destination and 2715.5 with a bf16 one, against 2843.6 and 2875.6 for the
// two-piece fit. v_if on the SFPU predicates, it does not branch, so the fit issued both
// of its pieces for every element; one formula is cheaper even with the error
// compensation below, which is a strictly serial chain and stalls more than its
// instruction count.
// =============================================================================
// Past this, 4·exp(-2|x|) is subnormal in fp32 and flushes to zero.
constexpr float TAIL_REGION_LIMIT = 45.0f;

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_tanh_derivative_sech2() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat result = 0.0f;

        // sech²(x) is an even function: sech²(-x) = sech²(x). Clear the sign bit rather
        // than take sfpi::abs, which leaves a sign-set NaN sign-set. That NaN then passes
        // the a < TAIL_REGION_LIMIT test below: -NaN came out as a finite ~1.8e-38 with an
        // fp32 destination and as +inf with a bf16 one, where +NaN gave 0. torch rounds
        // every NaN to bfloat16 as the sign-set 0xFFFF, so that was the common case.
        sfpi::vFloat a = sfpi::setsgn(val, 0);

        // At x = 0 the chain is exact: e = 1, (1 + e)² = 4, 4/4 = 1. With no region
        // boundary there is no step to come back up at. The fp32 result is not monotone
        // between adjacent inputs -- nothing short of correctly rounded is -- but every
        // step back up is exactly 1 ULP (exhaustive).
        //
        // Error compensation. The plain chain e4 · recip((1 + e)²) is 3 fp32 ULP at worst,
        // and the two roundings that dominate it are removed explicitly:
        //  * (1 + e)² is formed as e·(e + 2) + 1, one fused rounding, and that rounding is
        //    recovered as den_lo (together with the one in e + 2, via Fast2Sum) and carried
        //    to the end. Both subtractions are exact: (e + 2) - 2 by Sterbenz, and 1 - den
        //    because den ∈ [1, 4] sits on a grid no finer than 1's.
        //  * The quotient is rounded once, at the end: q0 = e4·y, then the residual
        //    e4 - q0·den - q0·den_lo is put back through y. That also absorbs the
        //    reciprocal's own error.
        // What is left is the exp's error, which the 742 inputs at 2 ULP are made of.
        // Carrying the exp's low part as well gets max 1 ULP everywhere, but measured at
        // roughly another 256 cycles/tile, which is slower than the two-piece fit it replaces.
        //
        // The a < TAIL_REGION_LIMIT guard is what sends the infinities and NaN of either
        // sign to 0.
        v_if(a < TAIL_REGION_LIMIT) {
            sfpi::vFloat e4 = inline_exp4_neg2x_fp32(a);  // 4·exp(-2|x|)
            sfpi::vFloat e = e4 * 0.25f;                  // exp(-2|x|), exact (power of two)
            sfpi::vFloat e2 = e + 2.0f;
            sfpi::vFloat e2_lo = e - (e2 - 2.0f);         // rounding error of e + 2
            sfpi::vFloat den = e * e2 + 1.0f;             // (1 + e)², one rounding
            sfpi::vFloat den_lo = e * e2 + (1.0f - den);  // its rounding error
            den_lo = e * e2_lo + den_lo;
            sfpi::vFloat y = inline_reciprocal_1_to_4(den);
            sfpi::vFloat q0 = e4 * y;
            sfpi::vFloat w = -q0 * den + e4;  // residual of q0 ...
            w = -q0 * den_lo + w;             // ... against den + den_lo
            result = w * y + q0;
        }
        v_endif;

        if constexpr (!is_fp32_dest_acc_en) {
            // Explicit RNE rounding for BF16 output — SFPSTORE truncates toward zero by default.
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void tanh_derivative_sech2_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    // No special initialization needed — no LUT, no programmable constants. The inline
    // exp uses only literals, and the reciprocal seeds from SFPARECIP (see
    // inline_reciprocal_1_to_4).
}

}  // namespace sfpu
}  // namespace ckernel
