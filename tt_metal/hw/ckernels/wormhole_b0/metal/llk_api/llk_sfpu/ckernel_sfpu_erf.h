// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

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
// FP32: n16/d16, 1 segment, range [-10.0, 10.0] (parity x²-Horner).
// ======================================================================

#ifdef INP_FLOAT32
constexpr std::uint32_t ERF_NUM_DEGREE = 16;
constexpr std::uint32_t ERF_DEN_DEGREE = 16;
constexpr std::uint32_t ERF_NUM_SEGMENTS = 1;
constexpr std::uint32_t ERF_LUT_SIZE = 36;
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

// n8/d8 rational (WH refit v2) — preserves MaxULP=1 vs truth; GELU-chain
// byte-match 99.3 % vs old polynomial GELU (v1: 99.0 %; baseline: 97.9 %).
// GELU-chain objective captures the composed error that CLIP PCC gate sees.
constexpr std::uint32_t ERF_NUM_DEGREE = 8;
constexpr std::uint32_t ERF_DEN_DEGREE = 8;
constexpr std::uint32_t ERF_NUM_SEGMENTS = 1;
constexpr std::uint32_t ERF_LUT_SIZE = 20;
constexpr std::array<float, ERF_LUT_SIZE> ERF_LUT = {
    {-1.0000000000e+01f, 1.0000000000e+01f, 0.0000000000e+00f, 1.1280932447e+00f, 0.0000000000e+00f,
     2.7609212279e-01f,  0.0000000000e+00f, 4.5400281738e-02f, 0.0000000000e+00f, 7.4481184425e-04f,
     0.0000000000e+00f,  1.0000000000e+00f, 0.0000000000e+00f, 5.7439188334e-01f, 0.0000000000e+00f,
     1.3675764810e-01f,  0.0000000000e+00f, 8.2844606784e-03f, 0.0000000000e+00f, 2.4813862145e-05f}};

#endif

// ======================================================================
// APPROXIMATION_MODE: 6-segment piecewise linear via SFPLUTFP32.
//
// erf is the ideal shape for the hardware LUT: odd, bounded, and already
// saturated before the last breakpoint, so segment 5 is the constant 1.0 and
// the explicit clamp the rational path needs is not required here.
//
// Segment boundaries on |x| are fixed in hardware (FP16 6-entry TABLE1, which
// is sfpi's default mode = 1): 0.5, 1.0, 1.5, 2.0, 3.0.
//
// The table is fitted on |x| and the sign is restored with copysgn. The natural
// spelling would be lut2() (SGN_RETAIN, which makes the hardware take the sign
// from the input), but sfpi 7.71.0's __builtin_rvtt_sfplutfp32_6r accepts only
// mod1 = 2 and 3 -- the two FP16 6-entry tables with SGN_UPDATE. SGN_RETAIN
// (mod1 | 4) is rejected at compile time, so the 6-entry LUT can only produce
// even functions and odd ones cost one SFPSETSGN. (This is also why tt-llk's
// 6-entry ckernel_sfpu_sigmoid.h, which calls lut2(), cannot build for WH.)
//
// Coefficients are IEEE fp16, packed lo/hi per imm32: slopes in LReg0/1/2 and
// intercepts in LReg4/5/6. A half-word of 0x7C00 reads as 0.0 (verified on
// device against the shipped GELU table, whose [3, inf) intercept uses it).
//
//   [0.0, 0.5): 1.067383*|x|
//   [0.5, 1.0): 0.644531*|x| + 0.213013
//   [1.0, 1.5): 0.246948*|x| + 0.604980
//   [1.5, 2.0): 0.058502*|x| + 0.881348
//   [2.0, 3.0): 0.004642*|x| + 0.987305
//   [3.0, inf): 1.0
//
// Segment 0's intercept is pinned to exactly 0 so that erf(0) == 0 and the fit
// stays continuous across the sign flip; a free fit puts 0.0082 there, which is
// a 2*0.0082 jump at the origin and sends small-|x| relative error to 3.6.
// Pinning costs nothing in absolute terms (0.01484 either way) and brings max
// relative error to 0.054.
//
// Max abs error 0.0148 over |x| <= 4 (~1.9 bf16 ULP at |y| ~ 1), against
// 0.0234 for the GELU table already shipping on this instruction.
// ======================================================================
template <int ITERATIONS>
inline void calculate_erf_appx() {
    sfpi::vUInt l0 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vUInt l1 = sfpi::l_reg[sfpi::LRegs::LReg1];
    sfpi::vUInt l2 = sfpi::l_reg[sfpi::LRegs::LReg2];
    sfpi::vUInt l4 = sfpi::l_reg[sfpi::LRegs::LReg4];
    sfpi::vUInt l5 = sfpi::l_reg[sfpi::LRegs::LReg5];
    sfpi::vUInt l6 = sfpi::l_reg[sfpi::LRegs::LReg6];

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = sfpi::copysgn(lut2_sign(x, l0, l1, l2, l4, l5, l6), x);
        sfpi::dst_reg++;
    }

    sfpi::l_reg[sfpi::LRegs::LReg0] = l0;
    sfpi::l_reg[sfpi::LRegs::LReg1] = l1;
    sfpi::l_reg[sfpi::LRegs::LReg2] = l2;
    sfpi::l_reg[sfpi::LRegs::LReg4] = l4;
    sfpi::l_reg[sfpi::LRegs::LReg5] = l5;
    sfpi::l_reg[sfpi::LRegs::LReg6] = l6;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_erf() {
    if constexpr (APPROXIMATION_MODE) {
        calculate_erf_appx<ITERATIONS>();
        return;
    }
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
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
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// Slopes (LReg0/1/2) and intercepts (LReg4/5/6) for the erf PWL table above.
// Shared with erfc, which evaluates 1 - erf via the same table.
inline void erf_appx_load_lut() {
    sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vUInt(0x39283C45);
    sfpi::l_reg[sfpi::LRegs::LReg1] = sfpi::vUInt(0x2B7D33E7);
    sfpi::l_reg[sfpi::LRegs::LReg2] = sfpi::vUInt(0x7C001CC1);
    sfpi::l_reg[sfpi::LRegs::LReg4] = sfpi::vUInt(0x32D17C00);
    sfpi::l_reg[sfpi::LRegs::LReg5] = sfpi::vUInt(0x3B0D38D7);
    sfpi::l_reg[sfpi::LRegs::LReg6] = sfpi::vUInt(0x3C003BE6);
}

template <bool APPROXIMATION_MODE>
void erf_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    if constexpr (APPROXIMATION_MODE) {
        erf_appx_load_lut();
    } else {
        sfpu_reciprocal_init<APPROXIMATION_MODE>();
    }
}

}  // namespace ckernel::sfpu
