// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"

#include "ckernel_sfpu_piecewise_rational.h"
#include "cmath_common.h"

namespace ckernel::sfpu {

// ======================================================================
// LUT-based erf via piecewise rational P(x)/Q(x)
//
// BF16: n8/d8, 1 segment, range [-10.0, 10.0] (parity x²-Horner)
// FP32: n16/d16, 1 segment, range [-10.0, 10.0] (parity x²-Horner)
// ======================================================================

#ifdef INP_FLOAT32
constexpr uint32_t ERF_NUM_DEGREE = 7;
constexpr uint32_t ERF_DEN_DEGREE = 7;
constexpr uint32_t ERF_NUM_SEGMENTS = 2;
constexpr uint32_t ERF_LUT_SIZE = 35;
constexpr std::array<float, ERF_LUT_SIZE> ERF_LUT = {{
    // Breakpoints
    0.0000000000e+00f,
    2.0000000000e+00f,
    4.0000000000e+00f,
    // Segment 0 [0, 2.0]: numerator (degree 7)
    0.0000000000e+00f,
    1.1283791673e+00f,
    -2.4294836001e-01f,
    1.1550096030e-01f,
    3.8071311477e-02f,
    1.4401851258e-02f,
    -1.0399653688e-03f,
    1.8679049155e-03f,
    // Segment 0 [0, 2.0]: denominator (degree 7)
    1.0000000000e+00f,
    -2.1530736409e-01f,
    4.3569306083e-01f,
    -3.8025706511e-02f,
    5.7972863184e-02f,
    8.0125357619e-03f,
    1.0333525153e-03f,
    1.6380826498e-03f,
    // Segment 1 [2.0, 4.0]: numerator (degree 7)
    2.4292787017e-01f,
    4.3090919808e-01f,
    -4.1881809714e-01f,
    3.6505715113e-01f,
    -1.0574575215e-01f,
    -8.4409561021e-03f,
    1.8356318795e-02f,
    -3.3304880063e-03f,
    // Segment 1 [2.0, 4.0]: denominator (degree 7)
    1.0000000000e+00f,
    -9.3909573160e-01f,
    6.4988625012e-01f,
    -1.0067531205e-01f,
    1.6677895050e-02f,
    -2.7846322937e-02f,
    2.0073265301e-02f,
    -3.3958820024e-03f}};

#else

// n8/d8 rational, coefficients aligned with WH v3 on-device refit (see PR #42540).
constexpr uint32_t ERF_NUM_DEGREE = 8;
constexpr uint32_t ERF_DEN_DEGREE = 8;
constexpr uint32_t ERF_NUM_SEGMENTS = 1;
constexpr uint32_t ERF_LUT_SIZE = 20;
constexpr std::array<float, ERF_LUT_SIZE> ERF_LUT = {
    {-1.0000000000e+01f, 1.0000000000e+01f, 0.0000000000e+00f, 1.1280932447e+00f, 0.0000000000e+00f,
     2.7609212279e-01f,  0.0000000000e+00f, 4.5400281738e-02f, 0.0000000000e+00f, 7.4481184425e-04f,
     0.0000000000e+00f,  1.0000000000e+00f, 0.0000000000e+00f, 5.7439188334e-01f, 0.0000000000e+00f,
     1.3675764810e-01f,  0.0000000000e+00f, 8.2844606784e-03f, 0.0000000000e+00f, 2.4813862145e-05f}};

#endif

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_erf() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
#ifdef INP_FLOAT32
        // Abs symmetry: erf(-x) = -erf(x). Clamp |x| to 4.0 before evaluation.
        // For |x| >= 4.0, erf(|x|) rounds to 1.0f in float32 (< 0.26 ULP error).
        sfpi::vFloat ax = sfpi::min(sfpi::abs(x), 4.0f);
        sfpi::vFloat result = piecewise_rational_eval<
            ERF_NUM_DEGREE,
            ERF_DEN_DEGREE,
            ERF_NUM_SEGMENTS,
            ERF_LUT_SIZE,
            false,
            APPROXIMATION_MODE>(ERF_LUT, ax);
        // Restore sign
        v_if(x < 0.0f) {
            result = -result;
        }
        v_endif;
#else
        // Clamp |x| to 10.0 before evaluation (erf is odd, rational is exact at boundary)
        x = sfpi::symmetric_clamp(x, 10.0f);
        sfpi::vFloat result = piecewise_rational_eval<
            ERF_NUM_DEGREE,
            ERF_DEN_DEGREE,
            ERF_NUM_SEGMENTS,
            ERF_LUT_SIZE,
            true,
            APPROXIMATION_MODE>(ERF_LUT, x);
        // Saturate to [-1, 1]: rational fit is not bounded and overshoots by
        // up to ~3e-8 (FP32) / ~2e-4 (BF16 LUT) in the tail. Persists in FP32
        // dest register and biases downstream ops (e.g. decomposed GELU in CLIP).
        result = sfpi::clamp(result, -1.0f, +1.0f);
#endif
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void erf_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
