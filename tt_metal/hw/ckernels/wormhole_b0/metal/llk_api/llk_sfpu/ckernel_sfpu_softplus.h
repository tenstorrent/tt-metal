// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpu/ckernel_sfpu_exp.h"
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
// BF16: degree-8 polynomial for f(a) on [0, 5] + inline exp tail
// FP32: same polynomial + inline exp + 3-term Taylor correction
// ======================================================================

constexpr float SOFTPLUS_POLY_BOUNDARY = 5.0f;

// Residual polynomial: f(a) = ln(1+exp(-a)) on [0, 5], degree 8
constexpr float SOFTPLUS_POLY_C0 = 6.9310557842e-01f;
constexpr float SOFTPLUS_POLY_C1 = -4.9926245213e-01f;
constexpr float SOFTPLUS_POLY_C2 = 1.2186349183e-01f;
constexpr float SOFTPLUS_POLY_C3 = 5.6753782555e-03f;
constexpr float SOFTPLUS_POLY_C4 = -1.0528374463e-02f;
constexpr float SOFTPLUS_POLY_C5 = 2.7290175203e-03f;
constexpr float SOFTPLUS_POLY_C6 = -3.4358495031e-04f;
constexpr float SOFTPLUS_POLY_C7 = 2.1285692128e-05f;
constexpr float SOFTPLUS_POLY_C8 = -4.8245715334e-07f;

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
        r, sfpi::vConst1, sfpi::vConst1, 0.5f, 0.166666667f, 0.0416666667f, 0.00833333333f, 0.00138888889f, 0.000198412698f);
#else
    // BF16: degree 5 sufficient
    sfpi::vFloat poly = PolynomialEvaluator::eval(r, sfpi::vConst1, sfpi::vConst1, 0.5f, 0.166666667f, 0.0416666667f, 0.00833333333f);
#endif

    // Scale by 2^k via exponent manipulation
    sfpi::vInt p_exp = sfpi::exexp(poly, sfpi::ExponentMode::NoDebias);
    sfpi::vInt new_exp = p_exp + k_int;

    // FTZ: if exponent underflows, result is 0
    sfpi::vFloat result = sfpi::vConst0;
    v_if(new_exp > 0) { result = sfpi::setexp(poly, new_exp); }
    v_endif;

    return result;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void calculate_softplus_body(const float beta, const float beta_reciprocal, const float threshold) {
    sfpi::vFloat val = sfpi::dst_reg[0];
    sfpi::vFloat t = beta * val;

    v_if(t < threshold) {
        // a = |t| via setsgn (clear sign bit, no branch)
        sfpi::vFloat a = sfpi::setsgn(t, 0);

        // f(a) via degree-8 Horner on [0, 5]
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

        // Tail: f(a) ≈ exp(-a) for a > 5
        sfpi::vFloat neg_a = sfpi::setsgn(a, 1);
        v_if(a > SOFTPLUS_POLY_BOUNDARY) {
#ifdef INP_FLOAT32
            // FP32: inline Cody-Waite exp + 3-term Taylor ln(1+e) = e*(1 + e*(-1/2 + e/3))
            sfpi::vFloat e = softplus_exp_negative(neg_a);
            residual = e * (sfpi::vConst1 + e * (-0.5f + e * 0.333333343f));
#else
            // BF16: exp_21f is faster than inline Cody-Waite (~8 vs ~15 ops)
            residual = _sfpu_exp_21f_bf16_<false>(neg_a);
#endif
        }
        v_endif;

        // Reconstruct softplus(t):
        //   t >= 0: softplus(t) = t + f(t) = max(0,t) + residual
        //   t < 0:  softplus(t) = f(|t|) = 0 + residual
        // Branch-free: vec_min_max clamps t to max(0,t), saving 1 instruction vs v_if
        sfpi::vFloat zero_threshold = 0.0f;
        sfpi::vec_min_max(zero_threshold, t);
        sfpi::vFloat sp = t + residual;

        // Round-to-nearest for bf16 destination (SFPSTORE defaults to truncation)
        sfpi::vFloat result = beta_reciprocal * sp;
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::float_to_fp16b(result, sfpi::RoundMode::NearestEven);
        }
        sfpi::dst_reg[0] = result;
    }
    v_endif;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_softplus(uint param0, uint param1, uint param2) {
    const float beta = Converter::as_float(param0);
    const float beta_reciprocal = Converter::as_float(param1);
    const float threshold = Converter::as_float(param2);
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_softplus_body<APPROXIMATION_MODE, is_fp32_dest_acc_en>(beta, beta_reciprocal, threshold);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void softplus_init() {}

}  // namespace ckernel::sfpu
