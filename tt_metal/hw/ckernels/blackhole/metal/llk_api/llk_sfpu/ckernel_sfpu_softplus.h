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
// BF16: single degree-8 polynomial for f(a) on [0, 5] + exp_21f tail
// FP32: two-segment shifted-variable polynomials on [0, 5] + Taylor tail
// ======================================================================

constexpr float SOFTPLUS_POLY_BOUNDARY = 5.0f;
constexpr float SOFTPLUS_SEG_MID = 2.5f;

// --- BF16 polynomial: f(a) on [0, 5], degree 8, raw variable a ---
constexpr float SOFTPLUS_BF16_C0 = 6.9310557842e-01f;
constexpr float SOFTPLUS_BF16_C1 = -4.9926245213e-01f;
constexpr float SOFTPLUS_BF16_C2 = 1.2186349183e-01f;
constexpr float SOFTPLUS_BF16_C3 = 5.6753782555e-03f;
constexpr float SOFTPLUS_BF16_C4 = -1.0528374463e-02f;
constexpr float SOFTPLUS_BF16_C5 = 2.7290175203e-03f;
constexpr float SOFTPLUS_BF16_C6 = -3.4358495031e-04f;
constexpr float SOFTPLUS_BF16_C7 = 2.1285692128e-05f;
constexpr float SOFTPLUS_BF16_C8 = -4.8245715334e-07f;

// --- FP32 polynomials: shifted variable u = (a - mid) / 1.25 ∈ [-1, 1] ---
// Segment 0: [0, 2.5], u = (a - 1.25) / 1.25, degree 10
constexpr float SOFTPLUS_FP32_S0_C0 = 2.5192907453e-01f;
constexpr float SOFTPLUS_FP32_S0_C1 = -2.7837514877e-01f;
constexpr float SOFTPLUS_FP32_S0_C2 = 1.3523818552e-01f;
constexpr float SOFTPLUS_FP32_S0_C3 = -3.1251531094e-02f;
constexpr float SOFTPLUS_FP32_S0_C4 = -6.8083574297e-04f;
constexpr float SOFTPLUS_FP32_S0_C5 = 2.6319222525e-03f;
constexpr float SOFTPLUS_FP32_S0_C6 = -5.4580852157e-04f;
constexpr float SOFTPLUS_FP32_S0_C7 = -1.3201816182e-04f;
constexpr float SOFTPLUS_FP32_S0_C8 = 8.3884675405e-05f;
constexpr float SOFTPLUS_FP32_S0_C9 = -1.9383776362e-06f;
constexpr float SOFTPLUS_FP32_S0_C10 = -6.0417200984e-06f;

// Segment 1: [2.5, 5], u = (a - 3.75) / 1.25, degree 9
constexpr float SOFTPLUS_FP32_S1_C0 = 2.3245465010e-02f;
constexpr float SOFTPLUS_FP32_S1_C1 = -2.8721714392e-02f;
constexpr float SOFTPLUS_FP32_S1_C2 = 1.7538610846e-02f;
constexpr float SOFTPLUS_FP32_S1_C3 = -6.9718970917e-03f;
constexpr float SOFTPLUS_FP32_S1_C4 = 1.9759770948e-03f;
constexpr float SOFTPLUS_FP32_S1_C5 = -3.9805736742e-04f;
constexpr float SOFTPLUS_FP32_S1_C6 = 4.6331253543e-05f;
constexpr float SOFTPLUS_FP32_S1_C7 = 3.5066736928e-06f;
constexpr float SOFTPLUS_FP32_S1_C8 = -3.8414204937e-06f;
constexpr float SOFTPLUS_FP32_S1_C9 = 9.6837720776e-07f;

template <bool APPROXIMATION_MODE>
inline void calculate_softplus_body(const float beta, const float beta_reciprocal, const float threshold) {
    sfpi::vFloat val = sfpi::dst_reg[0];
    sfpi::vFloat t = beta * val;

    v_if(t < threshold) {
        // a = |t|
        sfpi::vFloat a = t;
        v_if(t < 0.0f) { a = -t; }
        v_endif;

#ifdef INP_FLOAT32
        // FP32: two-segment shifted-variable polynomials for high precision
        // u = (a - segment_mid) / 1.25, maps each segment to [-1, 1]
        constexpr float INV_HW = 1.0f / 1.25f;  // = 0.8

        // Segment 0: [0, 2.5], u = (a - 1.25) * 0.8
        sfpi::vFloat u0 = (a - 1.25f) * INV_HW;
        sfpi::vFloat residual = PolynomialEvaluator::eval(
            u0,
            SOFTPLUS_FP32_S0_C0,
            SOFTPLUS_FP32_S0_C1,
            SOFTPLUS_FP32_S0_C2,
            SOFTPLUS_FP32_S0_C3,
            SOFTPLUS_FP32_S0_C4,
            SOFTPLUS_FP32_S0_C5,
            SOFTPLUS_FP32_S0_C6,
            SOFTPLUS_FP32_S0_C7,
            SOFTPLUS_FP32_S0_C8,
            SOFTPLUS_FP32_S0_C9,
            SOFTPLUS_FP32_S0_C10);

        // Segment 1: [2.5, 5], u = (a - 3.75) * 0.8
        v_if(a >= SOFTPLUS_SEG_MID) {
            sfpi::vFloat u1 = (a - 3.75f) * INV_HW;
            residual = PolynomialEvaluator::eval(
                u1,
                SOFTPLUS_FP32_S1_C0,
                SOFTPLUS_FP32_S1_C1,
                SOFTPLUS_FP32_S1_C2,
                SOFTPLUS_FP32_S1_C3,
                SOFTPLUS_FP32_S1_C4,
                SOFTPLUS_FP32_S1_C5,
                SOFTPLUS_FP32_S1_C6,
                SOFTPLUS_FP32_S1_C7,
                SOFTPLUS_FP32_S1_C8,
                SOFTPLUS_FP32_S1_C9);
        }
        v_endif;

        // Tail: Taylor series f(a) = e - e²/2 + e³/3 - e⁴/4
        v_if(a > SOFTPLUS_POLY_BOUNDARY) {
            sfpi::vFloat e = _sfpu_exp_accurate_<false>(a * -1.0f);
            sfpi::vFloat e2 = e * e;
            residual = e - e2 * 0.5f + e2 * e * 0.333333343f - e2 * e2 * 0.25f;
        }
        v_endif;
#else
        // BF16: single degree-8 polynomial + exp tail
        sfpi::vFloat residual = PolynomialEvaluator::eval(
            a,
            SOFTPLUS_BF16_C0,
            SOFTPLUS_BF16_C1,
            SOFTPLUS_BF16_C2,
            SOFTPLUS_BF16_C3,
            SOFTPLUS_BF16_C4,
            SOFTPLUS_BF16_C5,
            SOFTPLUS_BF16_C6,
            SOFTPLUS_BF16_C7,
            SOFTPLUS_BF16_C8);

        v_if(a > SOFTPLUS_POLY_BOUNDARY) { residual = _sfpu_exp_21f_bf16_<false>(-a); }
        v_endif;
#endif

        // Reconstruct softplus(t):
        //   t >= 0: softplus(t) = t + f(t)
        //   t < 0:  softplus(t) = f(|t|) = residual
        sfpi::vFloat sp = residual;
        v_if(t >= 0.0f) { sp = t + residual; }
        v_endif;

        sfpi::dst_reg[0] = beta_reciprocal * sp;
    }
    v_endif;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softplus(uint param0, uint param1, uint param2) {
    const float beta = Converter::as_float(param0);
    const float beta_reciprocal = Converter::as_float(param1);
    const float threshold = Converter::as_float(param2);
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_softplus_body<APPROXIMATION_MODE>(beta, beta_reciprocal, threshold);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void softplus_init() {
    _init_sfpu_reciprocal_<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
