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
// compatibility -- calculate_tanh_derivative_sech2 is accurate to 1 bfloat16 ULP instead.
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
// Inline exp for sech² tail: computes exp(-2|x| + ln4) = 4·exp(-2|x|)
// =============================================================================
// For |x| >= 3.0, the asymptotic approximation sech²(x) ≈ 4·exp(-2|x|) is
// exact to within 1 BF16 ULP (verified exhaustively).
//
// Uses the same inline Cody-Waite technique as x_times_exp_negative_tail()
// in the GELU backward kernel:
// 1. Range reduction: t = k·ln(2) + r, where k = round(t/ln2), |r| < ln(2)/2
// 2. Degree-4 Taylor polynomial for exp(r)
// 3. Direct exponent bit manipulation for 2^k scaling (FREE bit ops)
// 4. Explicit FTZ check via exponent field
//
// The ln4 trick folds the ×4 multiplication into the exp argument:
//   4·exp(-2|x|) = exp(-2|x| + ln4)
// This avoids FP32 FTZ underflow at |x| = 43.75 where exp(-87.5) < FP32 min
// normal but 4·exp(-87.5) ≈ 3.9e-38 is still a valid BF16 value.
//
// Performance: ~16 ops (vs ~28 for _sfpu_exp_f32_accurate_)
// Accuracy: < 1 ULP for the exp, combined with asymptotic formula gives Max ULP = 1
// =============================================================================
sfpi_inline sfpi::vFloat inline_exp_sech2_tail(sfpi::vFloat a) {
    constexpr float LN4 = 1.3862943611198906f;
    constexpr float INV_LN2 = 1.4426950408889634f;
    constexpr float LN2_HI = -0.6931152343750000f;  // -ln(2) high bits (exact in float)
    constexpr float LN2_LO = -3.19461832987e-05f;   // -ln(2) low bits

    // Taylor coefficients for exp(r), |r| < ln(2)/2 ≈ 0.347
    // Degree-4 is sufficient: the degree-5 term contributes < 0.0001 BF16 ULP
    // because 2^k attenuation in the tail (k = -7 to -125) shrinks the error
    // far below BF16 precision. Verified exhaustively on hardware: Max ULP = 1.
    constexpr float C2 = 0.5f;
    constexpr float C3 = 0.166666667f;
    constexpr float C4 = 0.0416666667f;

    // t = -2|x| + ln4 (always negative for |x| >= 3)
    sfpi::vFloat t = a * (-2.0f) + LN4;

    // Cody-Waite range reduction: t = k·ln(2) + r
    sfpi::vFloat z = t * INV_LN2;
    sfpi::vInt k_int;
    sfpi::vFloat k = _sfpu_round_to_nearest_int32_(z, k_int);

    sfpi::vFloat r = k * LN2_HI + t;  // Extended precision subtraction
    r = k * LN2_LO + r;

    // Degree-4 Taylor for exp(r)
    sfpi::vFloat poly = PolynomialEvaluator::eval(r, 1.0f, 1.0f, C2, C3, C4);

    // 2^k scaling via direct exponent bit manipulation (FREE)
    sfpi::vInt p_exp = sfpi::exexp(poly, sfpi::ExponentMode::Biased);
    sfpi::vInt new_exp = p_exp + k_int;

    // FTZ: if exponent underflows, result is 0 (natural zero saturation)
    sfpi::vFloat result = 0.0f;
    v_if(new_exp > 0) { result = sfpi::setexp(poly, new_exp); }
    v_endif;

    return result;
}

// =============================================================================
// Inline exp for the fp32-dest sech2 path: computes 4*exp(-2|x|)
// =============================================================================
// Same Cody-Waite shape as inline_exp_sech2_tail above, with constants chosen for an fp32
// destination. Two differences, both of them invisible in bfloat16 and both worth
// thousands of fp32 ulp:
//
// 1. The argument is -2|x| exactly (doubling never rounds) and the x4 is applied to
//    the reconstructed exponent rather than folded into the argument as +ln4.
//    Folding it in costs up to ulp(88)/2 = 3.8e-6 of absolute argument error at the
//    far end, which is 3.8e-6 of relative error in the result -- under a bfloat16
//    ulp, but ~60 fp32 ulp. Applying the x4 after the underflow test keeps what the
//    ln4 trick bought: 4*exp(-2|x|) is still a normal fp32 number out to |x| ~ 44.4,
//    well past the |x| = 43.7 where exp(-2|x|) alone goes subnormal and flushes.
// 2. The Taylor polynomial runs to degree 7. Degree 4 truncates at r^5/120 = 4.2e-5
//    relative for |r| <= ln(2)/2; degree 7 truncates at r^8/8! = 5.2e-9, under 0.1
//    fp32 ulp.
//
// k*LN2_HI is exact (LN2_HI is a 13-bit multiple of 2^-13 and |k| <= 128), and so is
// its sum with t, so the reduced argument carries only the LN2_LO rounding.
// =============================================================================
sfpi_inline sfpi::vFloat inline_exp4_neg2x_fp32(sfpi::vFloat a) {
    constexpr float INV_LN2 = 1.4426950408889634f;
    constexpr float LN2_HI = -0.6931152343750000f;  // -ln(2) high bits (exact in float)
    constexpr float LN2_LO = -3.19461832987e-05f;   // -ln(2) low bits

    constexpr float C2 = 0.5f;
    constexpr float C3 = 0.166666667f;
    constexpr float C4 = 0.0416666667f;
    constexpr float C5 = 0.00833333333f;
    constexpr float C6 = 0.00138888889f;
    constexpr float C7 = 1.98412698e-4f;

    sfpi::vFloat t = a * -2.0f;

    // Cody-Waite range reduction: t = k*ln(2) + r
    sfpi::vFloat z = t * INV_LN2;
    sfpi::vInt k_int;
    sfpi::vFloat k = _sfpu_round_to_nearest_int32_(z, k_int);

    sfpi::vFloat r = k * LN2_HI + t;
    r = k * LN2_LO + r;

    sfpi::vFloat poly = PolynomialEvaluator::eval(r, 1.0f, 1.0f, C2, C3, C4, C5, C6, C7);

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
// NaN guards in sfpu_reciprocal_iter are reachable. Two iterations from the SFPARECIP
// seed, matching the fp32 arm of _sfpu_reciprocal_gt0_ in ckernel_sfpu_trigonometry.h.
sfpi_inline sfpi::vFloat inline_reciprocal_1_to_4(sfpi::vFloat x) {
    sfpi::vFloat y = sfpi::approx_recip(x);
    sfpi::vFloat e = -x * y + 1.0f;
    y = y * e + y;
    e = -x * y + 1.0f;
    y = y * e + y;
    return y;
}

// =============================================================================
// Polynomial coefficients for sech²(x) = p(t), t = (2/9)·u - 1, u = x²
// =============================================================================
// Degree-10 minimax polynomial in scaled variable t ∈ [-1, 1], where t is
// derived from u = x² via t = (2/9)·u - 1.  Scaling to [-1, 1] keeps all
// powers bounded (|t^k| ≤ 1), eliminating the FP32 numerical instability
// that makes higher-degree unscaled polynomials diverge.
//
// Exploits even symmetry: sech²(-x) = sech²(x), so sech²(x) = f(x²) = p(t).
// Equivalent to degree-20 polynomial in x, with only 11 coefficients.
//
// Generated by generate_sech2_poly_scaled.py using weighted Chebyshev fitting
// with iterative Remez-style refinement.
// Validated in FP32 simulation: Max ULP = 1, 99.26% exact (ULP = 0).
//
// Evaluation: 1 MAD (scaling) + ~11 MAD ops (Horner) = ~12 MAD ops total.
// =============================================================================
constexpr float SECH2_POLY_C0 = 5.58971869398334972323e-02f;
constexpr float SECH2_POLY_C1 = -1.15151201975556879975e-01f;
constexpr float SECH2_POLY_C2 = 1.41578556434459129632e-01f;
constexpr float SECH2_POLY_C3 = -1.40162303416289407698e-01f;
constexpr float SECH2_POLY_C4 = 1.47692044971667296727e-01f;
constexpr float SECH2_POLY_C5 = -1.43049915574319674860e-01f;
constexpr float SECH2_POLY_C6 = 3.59505515570256450886e-02f;
constexpr float SECH2_POLY_C7 = 5.00465512420113917136e-02f;
constexpr float SECH2_POLY_C8 = 6.00316523798012785518e-02f;
constexpr float SECH2_POLY_C9 = -1.46346136132160548060e-01f;
constexpr float SECH2_POLY_C10 = 6.33840343077387569082e-02f;

// =============================================================================
// Accurate tanh derivative, one arm per destination format
// =============================================================================
// Both arms avoid the catastrophic cancellation in 1 - tanh²(x) (Max ULP = 15,140),
// and neither makes an external function call (_sfpu_exp_f32_accurate_,
// sfpu_reciprocal_iter).
//
// bf16 destination -- piecewise fit, same approach as the GELU backward kernel:
//   |x| < CORE_REGION_LIMIT:  Degree-10 minimax polynomial in t = (2/9)·a² - 1 (~12 MAD ops)
//   |x| >= CORE_REGION_LIMIT: Inline Cody-Waite exp(-2|x| + ln4) with FTZ (~16 ops)
//                             (zero saturation at |x| >= TAIL_REGION_LIMIT)
//   Accuracy: Max ULP = 1 across all 65,026 valid BF16 values (hardware verified).
//   Performance: ~14 ops (core) or ~18 ops (tail) vs ~40-50 ops (old version).
//
// fp32 destination -- the exact identity sech²(x) = 4e/(1 + e)², e = exp(-2|x|),
//   over the whole range, because the piecewise fit above is only ever within
//   1 bf16 ULP and that is 8.0e-4 relative at x = 0 and 5.0e-3 just past x = 3.
//   Accuracy: max 4 fp32 ULP, 69% correctly rounded and 94% within 1 ULP over a
//   65k-point Blackhole sweep of [-45, 45]; against 65,054 max ULP and 1.4%
//   correctly rounded for the bf16-grade arm on the same points.
//   Performance: ~30 ops.
//
// Both arms exploit even symmetry via a = |x|.
// =============================================================================
// Piecewise region boundaries for the bf16-destination sech²(x) approximation.
// Core region uses minimax polynomial; tail uses asymptotic exp formula;
// beyond tail, sech²(x) underflows BF16 min normal and saturates to zero. The
// fp32 arm uses TAIL_REGION_LIMIT only, as the point past which the result is
// subnormal in fp32 too.
constexpr float CORE_REGION_LIMIT = 3.0f;   // Polynomial ↔ exp boundary (bf16 arm)
constexpr float TAIL_REGION_LIMIT = 45.0f;  // Exp ↔ zero saturation boundary

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_tanh_derivative_sech2() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat result = 0.0f;

        // sech²(x) is an even function: sech²(-x) = sech²(x)
        sfpi::vFloat a = sfpi::abs(val);

        if constexpr (is_fp32_dest_acc_en) {
            // fp32 destination: the exact identity, not an approximation of the shape.
            //
            //   sech²(x) = 4e / (1 + e)²,  e = exp(-2|x|)
            //
            // The bf16 arm below is a two-piece fit whose pieces are each accurate to
            // 1 bf16 ULP and no further -- 8.0e-4 relative at x = 0, 5.0e-3 relative
            // just past x = 3, and a step *up* across the region boundary that makes a
            // strictly decreasing function come back non-monotone once the bf16 rounding
            // is removed. None of that is visible at bf16 output precision; all of it is
            // tens of thousands of fp32 ULP. See tenstorrent/tt-metal#57509.
            //
            // One formula over the whole range, so there is no boundary to step at. The
            // reciprocal argument is (1 + e)² ∈ [1, 4], and at x = 0 the chain is exact:
            // e = 1, (1 + e)² = 4, 4/4 = 1. Measured max 4 fp32 ULP and no monotonicity
            // violation over the full range on Blackhole.
            //
            // The a < TAIL_REGION_LIMIT guard is what sends the infinities and NaN to 0,
            // as in the bf16 arm; past it 4·exp(-2|x|) is subnormal in fp32 anyway.
            v_if(a < TAIL_REGION_LIMIT) {
                sfpi::vFloat e4 = inline_exp4_neg2x_fp32(a);  // 4·exp(-2|x|)
                sfpi::vFloat e = e4 * 0.25f;                  // exp(-2|x|), exact (power of two)
                sfpi::vFloat den = 1.0f + e;
                result = e4 * inline_reciprocal_1_to_4(den * den);
            }
            v_endif;
        } else {
            v_if(a < CORE_REGION_LIMIT) {
                // Core region: degree-10 polynomial in t = (2/9)·u - 1, u = a²
                // Scaling u ∈ [0, 9) → t ∈ [-1, 1) keeps powers bounded.
                sfpi::vFloat u = a * a;
                sfpi::vFloat t = u * (2.0f / 9.0f) + (-1.0f);
                result = PolynomialEvaluator::eval(
                    t,
                    SECH2_POLY_C0,
                    SECH2_POLY_C1,
                    SECH2_POLY_C2,
                    SECH2_POLY_C3,
                    SECH2_POLY_C4,
                    SECH2_POLY_C5,
                    SECH2_POLY_C6,
                    SECH2_POLY_C7,
                    SECH2_POLY_C8,
                    SECH2_POLY_C9,
                    SECH2_POLY_C10);
            }
            v_elseif(a < TAIL_REGION_LIMIT) {
                // Tail region: inline exp(-2|x| + ln4) = 4·exp(-2|x|)
                // Asymptotic formula exact to 1 BF16 ULP for |x| >= CORE_REGION_LIMIT.
                // Beyond TAIL_REGION_LIMIT, result stays 0 (sech²(45) ≈ 5.5e-39 < BF16 min normal).
                result = inline_exp_sech2_tail(a);
            }
            v_endif;

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
    // No special initialization needed — no reciprocal, no LUT.
    // Polynomial uses only Horner evaluation, inline exp uses only arithmetic.
}

}  // namespace sfpu
}  // namespace ckernel
