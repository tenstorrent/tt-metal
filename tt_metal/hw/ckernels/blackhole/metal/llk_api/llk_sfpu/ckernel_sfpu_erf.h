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
// BF16: n7/d8 odd rational, range [-10.0, 10.0], evaluated inline in x² (see below)
// FP32: n16/d16, 1 segment, range [-10.0, 10.0] (parity x²-Horner)
// ======================================================================

#ifdef INP_FLOAT32
constexpr uint32_t ERF_NUM_DEGREE = 16;
constexpr uint32_t ERF_DEN_DEGREE = 16;
constexpr uint32_t ERF_NUM_SEGMENTS = 1;
constexpr uint32_t ERF_LUT_SIZE = 36;
constexpr std::array<float, ERF_LUT_SIZE> ERF_LUT = {
    {-1.0000000000e+01f, 1.0000000000e+01f, 0.0000000000e+00f,  1.1283791065e+00f,  0.0000000000e+00f,
     2.1477432549e-01f,  0.0000000000e+00f, 6.2133435160e-02f,  0.0000000000e+00f,  5.6230435148e-03f,
     0.0000000000e+00f,  6.1307044234e-04f, 0.0000000000e+00f,  1.7678321456e-05f,  0.0000000000e+00f,
     2.7384647439e-08f,  0.0000000000e+00f, -2.8632063387e-10f, 0.0000000000e+00f,  1.0000000000e+00f,
     0.0000000000e+00f,  5.2367275953e-01f, 0.0000000000e+00f,  1.2961706519e-01f,  0.0000000000e+00f,
     1.9642570987e-02f,  0.0000000000e+00f, 1.9545555115e-03f,  0.0000000000e+00f,  1.3179056987e-04f,
     0.0000000000e+00f,  1.3156344494e-06f, 0.0000000000e+00f,  -3.5153888689e-09f, 0.0000000000e+00f,
     -6.7350725691e-12f}};

#else

// BF16 arm: odd rational x·P(x²)/Q(x²), P of degree 3 and Q of degree 4 in x², Q0 = 1. Two coefficient sets:
//
// - exact (APPROXIMATION_MODE = false, the ttnn default): P0 and Q1 are full fp32 constants held in
//   vConstFloatPrgm1/2 (programmed by erf_init); the other six are on the one-SFPLOADI fp16/bf16 grid. Grid-searched
//   against every bf16 input with both a BF16 (truncating store) and an FP32 dest so that max and mean ULP vs
//   float64 do not regress in any input range; P0 is 2/sqrt(pi) rounded to fp32. Blackhole only: this
//   intentionally departs from the WH v3 on-device refit (PR #42540) the previous set was aligned with.
// - fast_and_approx: the raw SFPARECIP (~7 bits) dominates the error, and the grid set measured worse on
//   silicon for |x| >= 4, so this mode keeps the WH-aligned set (P0 and Q1 still come from vConstFloatPrgm1/2).
template <bool APPROXIMATION_MODE>
struct ErfBf16Coeffs {
    static constexpr float P0 = 0x1.20dd76p+0f;  // vConstFloatPrgm1
    static constexpr float P1 = 0x1.1ap-2f;
    static constexpr float P2 = 0x1.738p-5f;
    static constexpr float P3 = 0x1.868p-11f;
    static constexpr float Q1 = 0x1.26425ap-1f;  // vConstFloatPrgm2
    static constexpr float Q2 = 0x1.18p-3f;
    static constexpr float Q3 = 0x1.0f8p-7f;
    static constexpr float Q4 = 0x1.9ep-16f;
};

template <>
struct ErfBf16Coeffs<true> {
    static constexpr float P0 = 1.1280932447e+00f;  // vConstFloatPrgm1
    static constexpr float P1 = 2.7609212279e-01f;
    static constexpr float P2 = 4.5400281738e-02f;
    static constexpr float P3 = 7.4481184425e-04f;
    static constexpr float Q1 = 5.7439188334e-01f;  // vConstFloatPrgm2
    static constexpr float Q2 = 1.3675764810e-01f;
    static constexpr float Q3 = 8.2844606784e-03f;
    static constexpr float Q4 = 2.4813862145e-05f;
};

#endif

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_erf() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
#ifdef INP_FLOAT32
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
#else
        // Same shape as the INP_FLOAT32 arm (clamp the signed x, odd rational, saturate to [-1, 1]), evaluated inline
        // so that P0 and Q1 come from vConstFloatPrgm1/2 (programmed by erf_init) instead of two SFPLOADI per row.
        using C = ErfBf16Coeffs<APPROXIMATION_MODE>;
        x = sfpi::symmetric_clamp(x, 10.0f);
        const sfpi::vFloat x2 = x * x;
        sfpi::vFloat den = C::Q4 * x2 + C::Q3;
        sfpi::vFloat num = C::P3 * x2 + C::P2;
        den = den * x2 + C::Q2;
        num = num * x2 + C::P1;
        den = den * x2 + sfpi::vConstFloatPrgm2;  // Q1
        num = num * x2 + sfpi::vConstFloatPrgm1;  // P0
        den = den * x2 + 1.0f;                    // Q0
        num = num * x;
        sfpi::vFloat result = num * sfpu_reciprocal<APPROXIMATION_MODE>(den);
        // Saturate to [-1, 1] (see the INP_FLOAT32 arm).
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
#ifndef INP_FLOAT32
    sfpi::vConstFloatPrgm1 = ErfBf16Coeffs<APPROXIMATION_MODE>::P0;
    sfpi::vConstFloatPrgm2 = ErfBf16Coeffs<APPROXIMATION_MODE>::Q1;
#endif
}

}  // namespace ckernel::sfpu
