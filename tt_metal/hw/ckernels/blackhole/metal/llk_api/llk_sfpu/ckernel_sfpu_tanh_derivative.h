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

// =============================================================================
// Accurate tanh derivative:  sech²(x) = 4e / (1 + e)²,  e = exp(-2|x|)
// =============================================================================
// The exact identity over the whole range, for both destination formats; a bfloat16
// destination only adds the final round-to-nearest-even. It replaces a two-piece
// bfloat16-grade fit (degree-10 polynomial below |x| = 3, exp tail above) that was tens of
// thousands of fp32 ULP off, stepped *up* across its region boundary, and misrounded 292
// bfloat16 results. See tenstorrent/tt-metal#57509.
//
// Blackhole p300a, exhaustive against mpmath:
//   fp32 dest:  max 2 ULP (742 of the 2.14e9 non-negative inputs), 99.0% correctly rounded,
//               adjacent inputs never step up by more than 1 ULP.
//   bf16 dest, and bf16 data through an fp32 dest:  all 65,280 finite inputs correctly rounded.
//   NaN and ±inf of either sign:  0.
// MATH_ISOLATE cycles/tile at ITERATIONS=32: 2369.5 (fp32 dest) / 2401.5 (bf16 dest), against
// 2843.6 / 2875.6 for the two-piece fit. v_if on the SFPU predicates rather than branches, so
// the fit paid for both of its pieces on every element.
// =============================================================================

// Past this 4·exp(-2|x|) is subnormal and flushes to zero; the guard on it is also what
// sends ±inf and NaN to 0, and what bounds |k| <= 130 in the range reduction below.
constexpr float SECH2_FLUSH_LIMIT = 45.0f;

// Cody-Waite split of -ln2, and the degree-6 minimax of _sfpu_exp_fp32_accurate_ for exp(r).
constexpr float SECH2_INV_LN2 = 1.4426950408889634f;
constexpr float SECH2_LN2_HI = -0.6931152343750000f;  // -ln2, high 13 bits
constexpr float SECH2_LN2_LO = -3.19461832987e-05f;   // -ln2 - SECH2_LN2_HI
constexpr float SECH2_C2 = 4.99999851e-1f, SECH2_C3 = 1.66664720e-1f, SECH2_C4 = 4.16695364e-2f;
constexpr float SECH2_C5 = 8.37312452e-3f, SECH2_C6 = 1.37805939e-3f;

// 4·exp(-2a) to fp32 accuracy for 0 <= a < SECH2_FLUSH_LIMIT.
//
// Cody-Waite reduction -2a = k·ln2 + r, the minimax above for exp(r) on |r| <= ln2/2, and
// 2^(k+2) written straight into the exponent field. k·LN2_HI is exact (a 13-bit constant
// times |k| <= 130) and so is its sum with -2a (Sterbenz), so r carries only the LN2_LO
// rounding. The ×4 goes into the exponent *before* the FTZ test, so the result stays normal
// out to a ≈ 44.4 where exp(-2a) alone is already subnormal; folding it into the argument as
// +ln4 instead would cost ~60 fp32 ULP at the far end.
//
// Every fp32 constant here costs two SFPLOADI per row if written as a literal, and the
// compiler does not hoist them. So the reduction constants come in as vFloats the caller
// loads once before its row loop, and C2..C4 sit in vConstFloatPrgm0..2 (programmed by
// tanh_derivative_sech2_init). C5 stays a literal: a fourth hoisted vFloat runs out of LRegs.
sfpi_inline sfpi::vFloat sech2_exp4_neg2x(
    sfpi::vFloat a, sfpi::vFloat inv_ln2, sfpi::vFloat ln2_hi, sfpi::vFloat ln2_lo) {
    sfpi::vFloat t = a * -2.0f;
    sfpi::vInt k_int;
    sfpi::vFloat k = _sfpu_round_to_nearest_int32_(t * inv_ln2, k_int);
    sfpi::vFloat r = k * ln2_hi + t;
    r = k * ln2_lo + r;
    sfpi::vFloat p = PolynomialEvaluator::eval(
        r, 1.0f, 1.0f, sfpi::vConstFloatPrgm0, sfpi::vConstFloatPrgm1, sfpi::vConstFloatPrgm2, SECH2_C5, SECH2_C6);

    sfpi::vInt e = sfpi::exexp(p, sfpi::ExponentMode::Biased) + k_int + 2;
    sfpi::vFloat result = 0.0f;
    v_if(e > 0) { result = sfpi::setexp(p, e); }  // FTZ
    v_endif;
    return result;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_tanh_derivative_sech2() {
    // Loop-invariant: loaded once here instead of two SFPLOADI each per row. Non-const on
    // purpose; a const vFloat live across the row loop fails to compile.
    sfpi::vFloat inv_ln2 = SECH2_INV_LN2;
    sfpi::vFloat ln2_hi = SECH2_LN2_HI;
    sfpi::vFloat ln2_lo = SECH2_LN2_LO;

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat result = 0.0f;

        // sech² is even. Clear the sign bit rather than take sfpi::abs, which leaves a
        // sign-set NaN sign-set and let it through the guard (-NaN gave 1.8e-38 with an fp32
        // destination and +inf with a bf16 one; torch writes every bf16 NaN as 0xFFFF).
        sfpi::vFloat a = sfpi::setsgn(val, 0);

        v_if(a < SECH2_FLUSH_LIMIT) {
            sfpi::vFloat e4 = sech2_exp4_neg2x(a, inv_ln2, ln2_hi, ln2_lo);  // 4e, exact at x = 0 (the whole chain is)

            // den = (1 + e)² = e4·q + 1 with q = (e + 2)/4, a single fused rounding. That
            // rounding and q's are recovered exactly -- 0.5 - q by Sterbenz, 1 - den because
            // den ∈ [1, 4] lies on a grid no finer than 1's -- and carried to the end as den_lo.
            sfpi::vFloat q = e4 * 0.0625f + 0.5f;
            sfpi::vFloat q_lo = e4 * 0.0625f + (0.5f - q);
            sfpi::vFloat den = e4 * q + 1.0f;
            sfpi::vFloat den_lo = e4 * q + (1.0f - den);
            den_lo = e4 * q_lo + den_lo;

            // 1/den, two Newton steps from the SFPARECIP seed as in _sfpu_reciprocal_gt0_<true>.
            // den ∈ [1, 4] with both ends attained (x = 0 gives 4, a flushed e4 gives 1), so
            // the seed needs no zero/inf/NaN guard.
            sfpi::vFloat y = sfpi::approx_recip(den);
            y = y * (-den * y + 1.0f) + y;
            y = y * (-den * y + 1.0f) + y;

            // Round the quotient once: q0 = e4·y, then the residual e4 - q0·(den + den_lo)
            // put back through y. What is left at 2 ULP is the exp's own rounding; carrying its
            // low part as well gets 1 ULP everywhere but costs ~256 cycles/tile more.
            sfpi::vFloat q0 = e4 * y;
            sfpi::vFloat w = -q0 * den + e4;
            w = -q0 * den_lo + w;
            result = w * y + q0;
        }
        v_endif;

        if constexpr (!is_fp32_dest_acc_en) {
            // SFPSTORE truncates; round to bfloat16 explicitly.
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void tanh_derivative_sech2_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    // The exp polynomial's C2..C4, read by sech2_exp4_neg2x. This takes all three programmable
    // constants, so sfpu_reciprocal_iter (which wants Prgm0 = 2.0) cannot be swapped in for the
    // SFPARECIP-seeded reciprocal without moving one of them back to a literal.
    sfpi::vConstFloatPrgm0 = SECH2_C2;
    sfpi::vConstFloatPrgm1 = SECH2_C3;
    sfpi::vConstFloatPrgm2 = SECH2_C4;
}

}  // namespace sfpu
}  // namespace ckernel
