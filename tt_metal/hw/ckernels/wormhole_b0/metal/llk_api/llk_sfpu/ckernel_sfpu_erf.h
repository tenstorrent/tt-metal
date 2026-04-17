// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"

#include "ckernel_sfpu_piecewise_rational.h"

namespace ckernel::sfpu {

// ======================================================================
// LUT-based erf via piecewise rational P(x)/Q(x)
//
// BF16: n8/d8, 1 segment, range [-10.0, 10.0] (parity x²-Horner).
//       WH-specific refit: coefficients re-optimized to maximize BF16-byte
//       match (~97.3 % on Gaussian inputs σ∈{0.5..3}) vs the pre-#41850
//       5-segment polynomial, while preserving MaxULP=1 vs FP64 truth at
//       BF16 output precision. Same kernel structure (same FMA count) as
//       pre-refit — perf neutral. See PR #42540 / plan 0089 for context:
//       WH SFPU FMA quirks on the 1-Newton-iter reciprocal path caused
//       CLIP encoder_2 to miss the 0.98 PCC gate; refit aligns output
//       bytes with the known-good polynomial pattern.
// FP32: n16/d16, 1 segment, range [-10.0, 10.0] (parity x²-Horner).
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

// n8/d8 rational (WH refit) — preserves MaxULP=1 vs truth; byte-matches
// the pre-#41850 5-segment polynomial on 97.3 % of Gaussian samples
// (baseline #41850 LUT: 94.1 %). Kernel ops and per-tile perf unchanged.
constexpr uint32_t ERF_NUM_DEGREE = 8;
constexpr uint32_t ERF_DEN_DEGREE = 8;
constexpr uint32_t ERF_NUM_SEGMENTS = 1;
constexpr uint32_t ERF_LUT_SIZE = 20;
constexpr std::array<float, ERF_LUT_SIZE> ERF_LUT = {
    {-1.0000000000e+01f, 1.0000000000e+01f, 0.0000000000e+00f, 1.1280128928e+00f, 0.0000000000e+00f,
     2.8960505579e-01f,  0.0000000000e+00f, 4.5152435537e-02f, 0.0000000000e+00f, 7.2198348185e-04f,
     0.0000000000e+00f,  1.0000000000e+00f, 0.0000000000e+00f, 5.8679490685e-01f, 0.0000000000e+00f,
     1.4200975465e-01f,  0.0000000000e+00f, 8.0304508532e-03f, 0.0000000000e+00f, 2.3856162623e-05f}};

#endif

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_erf() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        // Clamp |x| to 10.0 before evaluation (erf is odd, rational is exact at boundary)
        sfpi::vFloat ax = sfpi::setsgn(x, 0);
        sfpi::vFloat threshold = 10.0f;
        sfpi::vec_min_max(ax, threshold);
        x = sfpi::setsgn(ax, x);
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
        sfpi::vFloat neg_one = sfpi::vConstNeg1;
        sfpi::vFloat pos_one = sfpi::vConst1;
        sfpi::vec_min_max(neg_one, result);
        sfpi::vec_min_max(result, pos_one);
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void erf_init() {
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
