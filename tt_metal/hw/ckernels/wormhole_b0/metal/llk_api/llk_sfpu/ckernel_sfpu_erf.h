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
// BF16: n8/d8, 1 segment, range [-10.0, 10.0] (parity x²-Horner).
//       WH-specific refit (v3, on-device): coefficients re-optimized via
//       coordinate descent on actual WH SFPU hardware measurements.
//       On-device GELU byte-match vs pre-#41850 polynomial: 85.7 %
//       (Python model prediction was 99.3 % — WH FMA behavior diverges
//       significantly from Python IEEE FP32 model). MaxULP=1 vs FP64
//       truth preserved. Same kernel structure — perf neutral.
//       See PR #42540 / plan 0090.
// FP32: two polynomial pieces split at |x| = 1, input clamped to |x| <= 3.921875
// ======================================================================

#ifdef INP_FLOAT32
// FP32 (#58282): the domain is split in two, as erfc already does. With a = min(|x|, 3.921875)
// (erf rounds to 1.0f from |x| = 3.9192059; 3.921875 is exact in BF16) and the sign copied from x:
//   a < 1  : erf = a + a * P(a^2), P of degree 6, computed as 2^-24 * (s + s * P(a^2)) with
//            s = 2^24 * a; the power-of-two scalings are exact and keep s * P away from denormals;
//   a >= 1 : erf = Q(a - 2.46875), Q of degree 14.
// Checked on every positive FP32 input (erf is odd) with a bit-exact SFPMAD model of Blackhole and
// Wormhole: at most 1 ULP from erf computed in double precision. There is no reciprocal, so the
// approximate mode is as accurate as the exact one. Fit, model and exhaustive check:
// https://github.com/Maher75017/tt-erf-fp32
constexpr float ERF_INPUT_CLAMP = 3.921875f;
constexpr float ERF_SPLIT = 1.0f;
constexpr float ERF_CENTER = 2.46875f;
constexpr std::array<float, 7> ERF_P = {
    0x1.06eba8p-3f,
    -0x1.81273cp-2f,
    0x1.ce2d0cp-4f,
    -0x1.b7fb6cp-6f,
    0x1.5415d2p-8f,
    -0x1.a4564cp-11f,
    0x1.4a4bd4p-14f};
constexpr std::array<float, 15> ERF_Q = {
    0x1.ffc102p-1f,
    0x1.4d76c2p-9f,
    -0x1.9ba12p-8f,
    0x1.36fcbcp-7f,
    -0x1.3b3a72p-7f,
    0x1.b3912cp-8f,
    -0x1.7d36dap-9f,
    0x1.fab542p-12f,
    0x1.5cf298p-12f,
    -0x1.32f544p-12f,
    0x1.46f9d6p-14f,
    0x1.7f20f6p-16f,
    -0x1.217f12p-16f,
    0x1.fap-23f,
    0x1.238952p-20f};

#else

// n8/d8 rational (WH refit v2) — preserves MaxULP=1 vs truth; GELU-chain
// byte-match 99.3 % vs old polynomial GELU (v1: 99.0 %; baseline: 97.9 %).
// GELU-chain objective captures the composed error that CLIP PCC gate sees.
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
        sfpi::vFloat a = sfpi::min(sfpi::abs(x), ERF_INPUT_CLAMP);
        sfpi::vFloat u = a * a;
        sfpi::vFloat p = ERF_P[6];
#pragma GCC unroll 6
        for (int k = 5; k >= 0; k--) {
            p = p * u + ERF_P[k];
        }
        sfpi::vFloat s = a * 16777216.0f;                           // 2^24
        sfpi::vFloat inner = (s * p + s) * 5.9604644775390625e-8f;  // 2^-24
        sfpi::vFloat t = a - ERF_CENTER;
        sfpi::vFloat result = ERF_Q[14];
#pragma GCC unroll 14
        for (int k = 13; k >= 0; k--) {
            result = result * t + ERF_Q[k];
        }
        v_if(a < ERF_SPLIT) { result = inner; }
        v_endif;
        result = sfpi::copysgn(result, x);
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
#endif
        // Saturate to [-1, 1]: rational fit is not bounded and overshoots by
        // up to ~3e-8 (FP32) / ~2e-4 (BF16 LUT) in the tail. Persists in FP32
        // dest register and biases downstream ops (e.g. decomposed GELU in CLIP).
        result = sfpi::clamp(result, -1.0f, +1.0f);
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
