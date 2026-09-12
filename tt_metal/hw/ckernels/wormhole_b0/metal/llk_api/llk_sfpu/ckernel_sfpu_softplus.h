// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_polyval.h"

namespace ckernel::sfpu {

// ======================================================================
// Softplus via abs(x) symmetry + residual function
//
// Uses the identity: softplus(-x) = softplus(x) - x
// Defining f(a) = ln(1 + exp(-a)) for a >= 0:
//   softplus(t) = t + f(t)   for t >= 0
//   softplus(t) = f(-t)      for t < 0
//
// FP32: degree-8 polynomial for f(a) on [0, 5] + inline exp + 3-term Taylor tail
// BF16: degree-6 polynomial (bf16-accurate, <0.28 ULP) for t >= 0 (residual clamped to 0
//       past a > 5, since it is negligible next to the dominant t term there); for t < 0,
//       residual *is* the entire result, so the inline exp tail is used instead, bounded to
//       a < SOFTPLUS_EXP_TAIL_BOUNDARY where the tail helper's range reduction is valid
//       (see softplus_exp_negative below); beyond that the true value has already
//       underflowed to 0 in bf16, so clamping to 0 remains correct.
// ======================================================================

constexpr float SOFTPLUS_POLY_BOUNDARY = 5.0f;
// Upper bound on |t| for the inline exp tail (softplus_exp_negative): its range reduction
// via _sfpu_round_to_nearest_int32_ is only valid for |z| <= 2^22 (z = a * INV_LN2), i.e.
// roughly a <= 2.9e6. We use a much tighter bound of 88, since exp(-88) already underflows
// bf16 and fp32 to 0, so no accuracy is lost by clamping to 0 above it.
constexpr float SOFTPLUS_EXP_TAIL_BOUNDARY = 88.0f;

// FP32 residual polynomial: f(a) = ln(1+exp(-a)) on [0, 5], degree 8
constexpr float SOFTPLUS_POLY_C0 = 6.9310557842e-01f;
constexpr float SOFTPLUS_POLY_C1 = -4.9926245213e-01f;
constexpr float SOFTPLUS_POLY_C2 = 1.2186349183e-01f;
constexpr float SOFTPLUS_POLY_C3 = 5.6753782555e-03f;
constexpr float SOFTPLUS_POLY_C4 = -1.0528374463e-02f;
constexpr float SOFTPLUS_POLY_C5 = 2.7290175203e-03f;
constexpr float SOFTPLUS_POLY_C6 = -3.4358495031e-04f;
constexpr float SOFTPLUS_POLY_C7 = 2.1285692128e-05f;
constexpr float SOFTPLUS_POLY_C8 = -4.8245715334e-07f;

// BF16 residual polynomial: f(a) = ln(1+exp(-a)) on [0, 5], degree 6
// (ULP-weighted minimax fit; max error < 0.28 bf16 ULP over the domain)
constexpr float SOFTPLUS_BF16_POLY_C0 = 6.9423984729e-01f;
constexpr float SOFTPLUS_BF16_POLY_C1 = -5.0932420424e-01f;
constexpr float SOFTPLUS_BF16_POLY_C2 = 1.4279095486e-01f;
constexpr float SOFTPLUS_BF16_POLY_C3 = -1.3000584069e-02f;
constexpr float SOFTPLUS_BF16_POLY_C4 = -1.8627923291e-03f;
constexpr float SOFTPLUS_BF16_POLY_C5 = 5.0152968088e-04f;
constexpr float SOFTPLUS_BF16_POLY_C6 = -3.1273466851e-05f;

// ======================================================================
// Lightweight inline exp(x) for negative x (tail region).
// Adapted from gelu's x_times_exp_negative_tail (ckernel_sfpu_gelu.h).
// Uses Cody-Waite range reduction + Taylor polynomial.
// BF16: degree 5 (~15 ops), FP32: degree 7 (~19 ops).
// ======================================================================
sfpi_inline sfpi::vFloat softplus_exp_negative(sfpi::vFloat x) {
    constexpr float INV_LN2 = 1.4426950408889634f;
    constexpr float LN2_HI = -0.6931152343750000f;
    constexpr float LN2_LO = -3.19461832987e-05f;

    // Range reduction: x = k*ln(2) + r
    sfpi::vFloat z = x * INV_LN2;
    sfpi::vInt k_int;
    sfpi::vFloat k = _sfpu_round_to_nearest_int32_(z, k_int);

    // Cody-Waite: r = x - k*ln(2) in extended precision
    sfpi::vFloat r = k * LN2_HI + x;
    r = k * LN2_LO + r;

    // exp(r) via Taylor polynomial, |r| < 0.5
#ifdef INP_FLOAT32
    // FP32: degree 7 for < 1 ULP
    sfpi::vFloat poly = PolynomialEvaluator::eval(
        r, 1.0f, 1.0f, 0.5f, 0.166666667f, 0.0416666667f, 0.00833333333f, 0.00138888889f, 0.000198412698f);
#else
    // BF16: degree 5 sufficient
    sfpi::vFloat poly = PolynomialEvaluator::eval(r, 1.0f, 1.0f, 0.5f, 0.166666667f, 0.0416666667f, 0.00833333333f);
#endif

    // Scale by 2^k via exponent manipulation
    sfpi::vInt p_exp = sfpi::exexp(poly, sfpi::ExponentMode::Biased);
    sfpi::vInt new_exp = p_exp + k_int;

    // FTZ: if exponent underflows, result is 0
    sfpi::vFloat result = 0.0f;
    v_if(new_exp > 0) { result = sfpi::setexp(poly, new_exp); }
    v_endif;

    return result;
}

inline void softplus_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void calculate_softplus_body(const float beta, const float beta_reciprocal, const float threshold) {
    sfpi::vFloat val = sfpi::dst_reg[0];
    sfpi::vFloat t = beta * val;

    v_if(t <= threshold) {
        // a = |t| via setsgn (clear sign bit, no branch)
        sfpi::vFloat a = sfpi::setsgn(t, 0);

#ifdef INP_FLOAT32
        // FP32: f(a) via degree-8 Horner on [0, 5]
        sfpi::vFloat residual = PolynomialEvaluator::eval(
            a,
            SOFTPLUS_POLY_C0,
            SOFTPLUS_POLY_C1,
            SOFTPLUS_POLY_C2,
            SOFTPLUS_POLY_C3,
            SOFTPLUS_POLY_C4,
            SOFTPLUS_POLY_C5,
            SOFTPLUS_POLY_C6,
            SOFTPLUS_POLY_C7,
            SOFTPLUS_POLY_C8);

        // Tail: f(a) ≈ exp(-a) for a > 5, via inline Cody-Waite exp +
        // 3-term Taylor ln(1+e) = e*(1 + e*(-1/2 + e/3))
        v_if(a > SOFTPLUS_POLY_BOUNDARY) {
            sfpi::vFloat e = softplus_exp_negative(-a);
            residual = e * (1.0f + e * (-0.5f + e * 0.333333343f));
        }
        v_endif;
#else
        // BF16: f(a) via degree-6 Horner on [0, 5]
        sfpi::vFloat residual = PolynomialEvaluator::eval(
            a,
            SOFTPLUS_BF16_POLY_C0,
            SOFTPLUS_BF16_POLY_C1,
            SOFTPLUS_BF16_POLY_C2,
            SOFTPLUS_BF16_POLY_C3,
            SOFTPLUS_BF16_POLY_C4,
            SOFTPLUS_BF16_POLY_C5,
            SOFTPLUS_BF16_POLY_C6);

        // Tail: the degree-6 poly diverges past its [0, 5] fit domain, while the true
        // residual < exp(-5) = 0.0067 there. For t > 0, clamping residual to 0 is what
        // makes the reconstructed result equal t (see below): the true residual is below
        // bf16 rounding relative to t, so dropping it is a no-op in bf16 precision.
        // For t < 0, residual *is* the entire result (softplus(t) = 0 + residual), so it
        // must not be clamped to 0 there; use the inline exp tail with the same 3-term
        // ln(1+e) correction as the FP32 path above (raw exp(-a) alone is low by ~e/2,
        // about 1 bf16 ULP at a = 5), bounded to a < SOFTPLUS_EXP_TAIL_BOUNDARY (see
        // comment above) -- beyond that exp(-a) has already underflowed to 0, and calling
        // the tail there would feed values outside its valid range reduction domain.
        v_if(a > SOFTPLUS_POLY_BOUNDARY) {
            residual = 0.0f;
            v_if(t < 0.0f && a < SOFTPLUS_EXP_TAIL_BOUNDARY) {
                sfpi::vFloat e = softplus_exp_negative(-a);
                residual = e * (1.0f + e * (-0.5f + e * 0.333333343f));
            }
            v_endif;
        }
        v_endif;
#endif

        // Reconstruct softplus(t):
        //   t >= 0: softplus(t) = t + f(t) = max(0,t) + residual
        //   t < 0:  softplus(t) = f(|t|) = 0 + residual
        t = sfpi::max(t, 0.0f);
        sfpi::vFloat sp = t + residual;

        // Round-to-nearest for bf16 destination (SFPSTORE defaults to truncation)
        sfpi::vFloat result = beta_reciprocal * sp;
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
    }
    v_endif;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_softplus(std::uint32_t param0, std::uint32_t param1, std::uint32_t param2) {
    const float beta = Converter::as_float(param0);
    const float beta_reciprocal = Converter::as_float(param1);
    const float threshold = Converter::as_float(param2);
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_softplus_body<APPROXIMATION_MODE, is_fp32_dest_acc_en>(beta, beta_reciprocal, threshold);
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
