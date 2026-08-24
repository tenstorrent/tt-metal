// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_converter.h"

#include "ckernel_sfpu_piecewise_rational.h"
#ifdef INP_FLOAT32
#include "ckernel_sfpu_exp.h"  // For _sfpu_round_to_nearest_int32_
#include "sfpu/ckernel_sfpu_polyval.h"
#endif

namespace ckernel::sfpu {

// ======================================================================
// LUT-based erfc via piecewise rational P(x)/Q(x)
//
// Uses abs(x) symmetry: erfc(-x) = 2 - erfc(x)
// Fit on [0, 5.0] only, 2 segments with n4/d5 rational per segment.
// BF16 MaxULP=118 (was 128 with 3-seg n4/d4 on [-5,5])
//
// FP32: the shared 2-segment fit below is BF16-only as of this change. FP32
// used the same n4/d5 rational for segment 1 [2.5, 5.0] and was catastrophically
// ill-conditioned there -- erfc(x) spans ~9 orders of magnitude over that segment
// (0.0004 down to 1.5e-12), which a low-degree rational cannot represent to
// relative accuracy (self-documented "FP32 MaxULP~9M", independently reproduced
// at 9,387,962 ULP / 114% relative error near x=5 -- issue #54053). See the
// INP_FLOAT32 branch of calculate_erfc() below for the fix: segment 0 [0, 2.5]
// is unchanged (already reasonable for FP32, ~2247 max ULP); segment 1 is
// replaced with the standard asymptotic decomposition erfc(x) = exp(-x^2) *
// corr(x), the same H(x) = exp(...) * corr_H(x) idiom this codebase already
// uses for gelu.h's negative tail -- corr(x) is smooth and bounded ([0.11,
// 0.21] over this range) so a modest-degree polynomial fits it to high
// relative accuracy, unlike erfc(x) itself.
// 18 FMAs          (was 24)
// ======================================================================

constexpr uint32_t ERFC_NUM_DEGREE = 4;
constexpr uint32_t ERFC_DEN_DEGREE = 5;
constexpr uint32_t ERFC_NUM_SEGMENTS = 2;
constexpr uint32_t ERFC_LUT_SIZE = 25;
constexpr std::array<float, ERFC_LUT_SIZE> ERFC_LUT = {{// Breakpoints
                                             0.0000000000e+00f,
                                             2.5000000000e+00f,
                                             5.0000000000e+00f,
                                             // Segment 0 [0, 2.5]: numerator (degree 4)
                                             1.0000233650e+00f,
                                             -1.3375675678e+00f,
                                             6.8185544014e-01f,
                                             -1.5691982210e-01f,
                                             1.3746744953e-02f,
                                             // Segment 0 [0, 2.5]: denominator (degree 5)
                                             1.0000000000e+00f,
                                             -2.0801517367e-01f,
                                             4.3667086959e-01f,
                                             -3.4568668343e-03f,
                                             2.5104774162e-02f,
                                             2.8375532478e-02f,
                                             // Segment 1 [2.5, 5.0]: numerator (degree 4)
                                             -2.5655237550e-05f,
                                             2.1275576728e-05f,
                                             -6.6162156145e-06f,
                                             9.1439767402e-07f,
                                             -4.7387182178e-08f,
                                             // Segment 1 [2.5, 5.0]: denominator (degree 5)
                                             1.0000000000e+00f,
                                             -1.6457208991e-01f,
                                             -2.0572184026e-01f,
                                             -1.3888636231e-01f,
                                             1.2677097321e-01f,
                                             -2.1375391632e-02f}};

#ifdef INP_FLOAT32
// FP32-accurate path (issue #54053 fix). BF16 mode above is untouched -- the
// self-documented BF16 MaxULP=118 was already acceptable; the catastrophic
// failure was FP32-only.

// Segment [0, 2.5]: reuses the existing n4/d5 rational fit's coefficients
// directly (unaffected by this fix -- already reasonable for FP32: max 2247
// ULP, mean 201 ULP over this sub-range, evaluated with a 2-Newton-Raphson-step
// reciprocal below instead of the shared LUT path's 0-step approximate
// reciprocal, so this is a slight accuracy improvement over the prior FP32
// behavior, not a regression).
constexpr float ERFC_FP32_SEG0_NUM[5] = {
    1.0000233650e+00f, -1.3375675678e+00f, 6.8185544014e-01f, -1.5691982210e-01f, 1.3746744953e-02f};
constexpr float ERFC_FP32_SEG0_DEN[6] = {
    1.0000000000e+00f,
    -2.0801517367e-01f,
    4.3667086959e-01f,
    -3.4568668343e-03f,
    2.5104774162e-02f,
    2.8375532478e-02f};

// Segment (2.5, 5.0]: erfc(x) = exp(-x^2) * corr(x), the exponential-envelope
// decomposition. Cody-Waite range reduction (identical constants to
// gelu.h's x_times_exp_negative_tail) computes exp(-x^2); no underflow guard
// is needed here (exp(-25) at x=5 is a normal FP32 value, unlike gelu's tail
// which reaches subnormal-adjacent magnitudes).
constexpr float ERFC_TAIL_INV_LN2 = 1.4426950408889634f;
constexpr float ERFC_TAIL_LN2_HI = -0.6931152343750000f;
constexpr float ERFC_TAIL_LN2_LO = -3.19461832987e-05f;

// corr(u), u = |x| - 3.75 (segment midpoint). Centering the Horner evaluation
// on the segment midpoint -- rather than evaluating in raw x -- dropped max
// ULP from 153 to 19 in validation, by avoiding FP32 rounding accumulation
// over the [2.5, 5.0] range; the coefficients' own double-precision fit
// residual (1.7e-7 relative) was already far smaller than that gap.
// Ascending order (c0 + c1*u + c2*u^2 + ...), degree 8.
constexpr float ERFC_TAIL_CORR[9] = {
    1.4558972e-01f,
    -3.6456216e-02f,
    8.8787700e-03f,
    -2.1076668e-03f,
    4.8821830e-04f,
    -1.0962165e-04f,
    2.4412318e-05f,
    -6.1850087e-06f,
    1.2795764e-06f};

// Validated: bit-exact FP32 model (plain sequential Horner, matching the
// SplitThreshold=99 pin below) vs scipy.special.erfc over 2,000,001 points
// in [2.5, 5.0] -- max ULP 19, mean ULP 3.9 (down from max 9,387,962 / 114%
// relative error). See sfpu_audit/fix_erfc/fit_and_validate.py. Not yet
// validated on real SFPU hardware -- the round-to-nearest-int32 bit-trick
// below (_sfpu_round_to_nearest_int32_) is expected to match the plain
// round() used in the Python model for this input range, but that
// equivalence itself has not been hardware-checked.
#endif  // INP_FLOAT32

template <int ITERATIONS = 8>
inline void calculate_erfc() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        // Clamp |x| to 5.0 before evaluation (avoids extrapolation, saves one branch)
        sfpi::vFloat ax = sfpi::min(sfpi::abs(x), 5.0f);
        sfpi::vFloat r;
#ifdef INP_FLOAT32
        v_if(ax > 2.5f) {
            sfpi::vFloat t = -(ax * ax);
            sfpi::vFloat z = t * ERFC_TAIL_INV_LN2;
            sfpi::vInt k_int;
            sfpi::vFloat k = _sfpu_round_to_nearest_int32_(z, k_int);

            sfpi::vFloat red = k * ERFC_TAIL_LN2_HI + t;
            red = k * ERFC_TAIL_LN2_LO + red;

            // Degree-8 Taylor exp(red); SplitThreshold pinned above the
            // coefficient count to force plain Horner, matching the Python
            // validation model bit-for-bit (the default even/odd split
            // scheme rounds slightly differently in the last bits).
            sfpi::vFloat poly = PolynomialEvaluator::eval<99>(
                red, 1.0f, 1.0f, 0.5f, 0.166666667f, 0.0416666667f, 0.00833333333f, 0.00138888889f,
                0.000198412698f, 0.0000248015873f);

            sfpi::vInt exp_biased = sfpi::exexp(poly, sfpi::ExponentMode::Biased);
            sfpi::vInt new_exp = exp_biased + k_int;
            sfpi::vFloat exp_val = sfpi::setexp(poly, new_exp);

            sfpi::vFloat u = ax - 3.75f;
            sfpi::vFloat corr = PolynomialEvaluator::eval<99>(
                u,
                ERFC_TAIL_CORR[0],
                ERFC_TAIL_CORR[1],
                ERFC_TAIL_CORR[2],
                ERFC_TAIL_CORR[3],
                ERFC_TAIL_CORR[4],
                ERFC_TAIL_CORR[5],
                ERFC_TAIL_CORR[6],
                ERFC_TAIL_CORR[7],
                ERFC_TAIL_CORR[8]);

            r = exp_val * corr;
        }
        v_else {
            sfpi::vFloat numer, denom;
            piecewise_rational_eval_numer_denom<4, 5>(ERFC_FP32_SEG0_NUM, ERFC_FP32_SEG0_DEN, ax, numer, denom);
            r = numer * sfpu_reciprocal<false>(denom);
        }
        v_endif;
#else
        r = piecewise_rational_eval<ERFC_NUM_DEGREE, ERFC_DEN_DEGREE, ERFC_NUM_SEGMENTS, ERFC_LUT_SIZE, false, true>(
            ERFC_LUT, ax);
#endif
        // erfc(-x) = 2 - erfc(x)
        v_if(x < 0.0f) { r = 2.0f - r; }
        v_endif;
        sfpi::dst_reg[0] = r;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void erfc_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<true>();
}

}  // namespace ckernel::sfpu
