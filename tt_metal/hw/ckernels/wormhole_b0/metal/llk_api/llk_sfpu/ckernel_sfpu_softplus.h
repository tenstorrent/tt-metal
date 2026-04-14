// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpu/ckernel_sfpu_exp.h"

// Adaptive per-segment degree — reduces Horner steps for low-degree segments
#define HAS_SEGMENT_DEGREES
#ifdef INP_FLOAT32
constexpr uint32_t SEGMENT_DEGREES[] = {11, 11};
#else
constexpr uint32_t SEGMENT_DEGREES[] = {8, 8, 6, 4};
#endif

#include "ckernel_sfpu_piecewise_polynomial.h"
#include "ckernel_sfpu_piecewise_rational.h"

namespace ckernel::sfpu {

// ======================================================================
// LUT-based softplus via abs(x) symmetry + residual function
//
// Uses the identity: softplus(-x) = softplus(x) - x
// Equivalently, defining f(a) = ln(1 + exp(-a)) for a >= 0:
//   softplus(t) = t + f(t)   for t >= 0
//   softplus(t) = f(-t)      for t < 0
//
// The LUT approximates f(a) on [0, LUT_HI], halving the segment count.
//
// BF16: polynomial, 4 segment(s), range [0, 10.0], max degree 8
// FP32: rational n11/d11, 2 segment(s), range [-10.0, 10.0]
// ======================================================================

#ifdef INP_FLOAT32
// FP32 path: rational n11/d11 on [-10, 10] (original PR LUT, uses full-range eval)
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

#else
// BF16 path: polynomial approximation of f(a) = ln(1 + exp(-a)) on [0, 12]
// 4 segments, max degree 8, 41 LUT entries (vs 241 for the full-range 16-segment version)
// Range [0, 12] ensures f(a) covers all bf16-representable residuals (exp(-12) ≈ 6e-6)
constexpr uint32_t SOFTPLUS_NUM_DEGREE = 8;
constexpr uint32_t SOFTPLUS_NUM_SEGMENTS = 4;
constexpr uint32_t SOFTPLUS_LUT_SIZE = 41;
constexpr float SOFTPLUS_LUT_HI = 12.0f;
constexpr std::array<float, 41> SOFTPLUS_LUT = {
    {0.0000000000e+00f,  3.0000000000e+00f,  6.0000000000e+00f,  9.0000000000e+00f,  1.2000000000e+01f,
     6.9314652681e-01f,  -4.9998274446e-01f, 1.2489502877e-01f,  2.4864752777e-04f,  -5.4577449337e-03f,
     4.2756633775e-05f,  4.7246625763e-04f,  -1.1166145123e-04f, 8.5480596681e-06f,  6.4356017113e-01f,
     -4.1999831796e-01f, 6.9007471204e-02f,  2.4819521233e-02f,  -1.4719141647e-02f, 3.3421402331e-03f,
     -4.1296074050e-04f, 2.7464138839e-05f,  -7.7396288134e-07f, 3.3762899041e-01f,  -2.0769727230e-01f,
     5.4375723004e-02f,  -7.7192140743e-03f, 6.2426319346e-04f,  -2.7179625249e-05f, 4.9635303867e-07f,
     0.0000000000e+00f,  0.0000000000e+00f,  2.2023772821e-02f,  -7.4348580092e-03f, 9.4883231213e-04f,
     -5.4178526625e-05f, 1.1665415514e-06f,  0.0000000000e+00f,  0.0000000000e+00f,  0.0000000000e+00f,
     0.0000000000e+00f}};

#endif

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
        // BF16: symmetry approach — evaluate f(a) = ln(1+exp(-a)) on [0, LUT_HI]
        // then reconstruct: softplus(t) = t + f(|t|) if t >= 0, f(|t|) if t < 0

        // Compute a = |t| via conditional negate (SFPI has no abs instruction)
        sfpi::vFloat a = t;
        v_if(t < 0.0f) { a = -t; }
        v_endif;

        // Evaluate residual f(a) from LUT
        sfpi::vFloat residual =
            piecewise_polynomial_eval<SOFTPLUS_NUM_DEGREE, SOFTPLUS_NUM_SEGMENTS, SOFTPLUS_LUT_SIZE>(SOFTPLUS_LUT, a);

        // Tail: f(a) ≈ exp(-a) for a > LUT_HI
        v_if(a > SOFTPLUS_LUT_HI) { residual = _sfpu_exp_21f_bf16_<false>(-a); }
        v_endif;

        // Reconstruct softplus(t):
        //   t >= 0: softplus(t) = t + f(t)
        //   t < 0:  softplus(t) = f(-t) = f(a) = residual
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
