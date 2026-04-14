// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpu/ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_polyval.h"

#ifdef INP_FLOAT32
#define HAS_SEGMENT_DEGREES
constexpr uint32_t SEGMENT_DEGREES[] = {11, 11};
#include "ckernel_sfpu_piecewise_polynomial.h"
#include "ckernel_sfpu_piecewise_rational.h"
#endif

namespace ckernel::sfpu {

// ======================================================================
// LUT-based softplus via abs(x) symmetry + residual function
//
// Uses the identity: softplus(-x) = softplus(x) - x
// Defining f(a) = ln(1 + exp(-a)) for a >= 0:
//   softplus(t) = t + f(t)   for t >= 0
//   softplus(t) = f(-t)      for t < 0
//
// BF16: single degree-8 polynomial for f(a) on [0, 5] + exp(-a) tail
// FP32: rational n11/d11 on [-10, 10] (PR's original LUT)
// ======================================================================

#ifdef INP_FLOAT32
// FP32 path: rational n11/d11 on [-10, 10] (unchanged from PR)
constexpr uint32_t SOFTPLUS_NUM_DEGREE = 11;
constexpr uint32_t SOFTPLUS_DEN_DEGREE = 11;
constexpr uint32_t SOFTPLUS_NUM_SEGMENTS = 2;
constexpr uint32_t SOFTPLUS_LUT_SIZE = 51;
constexpr float SOFTPLUS_LUT_LO = -10.0f;
constexpr float SOFTPLUS_LUT_HI = 10.0f;
constexpr std::array<float, 51> SOFTPLUS_LUT = {
    {-1.0000000000e+01f, 0.0000000000e+00f,  1.0000000000e+01f,  6.9314718246e-01f,  -1.5993961543e+00f,
     -2.5127007067e-01f, -1.3265351206e-01f, -2.9783745878e-02f, -1.5730193118e-03f, 2.8420342278e-04f,
     5.4250589073e-05f,  4.2192577894e-06f,  1.8079879105e-07f,  4.2213262608e-09f,  4.2415547203e-11f,
     1.0000000000e+00f,  -3.0287885070e+00f, 1.6419652700e+00f,  -8.2960790396e-01f, 2.6686237752e-01f,
     -6.7941252142e-02f, 1.3101734454e-02f,  -1.8655313179e-03f, 1.8875836395e-04f,  -2.0982502519e-05f,
     7.1359330178e-07f,  -9.3163990300e-08f, 6.9314718246e-01f,  7.5465214074e+01f,  9.6088657975e+01f,
     6.4373108774e+01f,  2.9242373943e+01f,  9.6345350314e+00f,  2.3334935121e+00f,  4.0709061723e-01f,
     4.8712256554e-02f,  3.5111696889e-03f,  3.2350137371e-04f,  -1.5047511503e-06f, 1.0000000000e+00f,
     1.0815194273e+02f,  6.0431155443e+01f,  2.9775133565e+01f,  9.8190872483e+00f,  2.2600082848e+00f,
     4.1877857578e-01f,  4.7625164822e-02f,  3.5753458696e-03f,  3.2110606327e-04f,  -1.4526502952e-06f,
     -5.0736897138e-10f}};
#endif

// BF16 residual polynomial: f(a) = ln(1+exp(-a)) on [0, 5], degree 8
// For a > 5, exp(-a) is used instead (seamless transition in bf16).
constexpr float SOFTPLUS_POLY_BOUNDARY = 5.0f;
constexpr float SOFTPLUS_POLY_C0 = 6.9310557842e-01f;
constexpr float SOFTPLUS_POLY_C1 = -4.9926245213e-01f;
constexpr float SOFTPLUS_POLY_C2 = 1.2186349183e-01f;
constexpr float SOFTPLUS_POLY_C3 = 5.6753782555e-03f;
constexpr float SOFTPLUS_POLY_C4 = -1.0528374463e-02f;
constexpr float SOFTPLUS_POLY_C5 = 2.7290175203e-03f;
constexpr float SOFTPLUS_POLY_C6 = -3.4358495031e-04f;
constexpr float SOFTPLUS_POLY_C7 = 2.1285692128e-05f;
constexpr float SOFTPLUS_POLY_C8 = -4.8245715334e-07f;

template <bool APPROXIMATION_MODE>
inline void calculate_softplus_body(const float beta, const float beta_reciprocal, const float threshold) {
    sfpi::vFloat val = sfpi::dst_reg[0];
    sfpi::vFloat t = beta * val;

    v_if(t < threshold) {
#ifdef INP_FLOAT32
        // FP32: rational eval on full range [-10, 10], with boundary clamps
        sfpi::vFloat result =
            piecewise_rational_eval<SOFTPLUS_NUM_DEGREE, SOFTPLUS_DEN_DEGREE, SOFTPLUS_NUM_SEGMENTS, SOFTPLUS_LUT_SIZE>(
                SOFTPLUS_LUT, t);
        v_if(t > SOFTPLUS_LUT_HI) { result = t; }
        v_endif;
        v_if(t < SOFTPLUS_LUT_LO) { result = 0.0f; }
        v_endif;
        sfpi::dst_reg[0] = beta_reciprocal * result;
#else
        // BF16: symmetry approach with inline Horner + exp tail
        // a = |t|
        sfpi::vFloat a = t;
        v_if(t < 0.0f) { a = -t; }
        v_endif;

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

        // Tail: f(a) ≈ exp(-a) for a > 5 (seamless in bf16)
        v_if(a > SOFTPLUS_POLY_BOUNDARY) { residual = _sfpu_exp_21f_bf16_<false>(-a); }
        v_endif;

        // Reconstruct softplus(t):
        //   t >= 0: softplus(t) = t + f(t)
        //   t < 0:  softplus(t) = f(|t|) = residual
        sfpi::vFloat sp = residual;
        v_if(t >= 0.0f) { sp = t + residual; }
        v_endif;

        sfpi::dst_reg[0] = beta_reciprocal * sp;
#endif
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
