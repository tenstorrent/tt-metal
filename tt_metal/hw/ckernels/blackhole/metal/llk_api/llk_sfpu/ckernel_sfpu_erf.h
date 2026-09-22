// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
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

// ======================================================================
// Fast bf16 erf for Blackhole. Origin: llk-bench (LLM-agent-written kernel, claude-fable-5, 2026-08),
// validated exhaustively over all 65,536 bf16 inputs vs the exact golden: max 2 ULP (gate <= 2).
// Measured 626.1 cycles/tile vs 1519.1 for the previous kernel on p150b (tt-metal v0.76.0 baseline).
//
// Pure sfpi: no SFPLOADMACRO, no replay slots. Init programs the four programmable constant registers
// LREG11 (vConstNeg1 := c2, i.e. NOT -1 while this op is active), LREG12 (c1), LREG13 (c0), LREG14 (S).
// bf16 DEST only: the truncating bf16 SFPSTORE is part of the algorithm (coefficients pre-scaled).
// Selected by calculate_erf / erf_init when !APPROXIMATION_MODE && !is_fp32_dest_acc_en (&& ITERATIONS == 8).
// ======================================================================

// deg-3 fit on t in [0,4], scaled by SIGMA = 1.0045 (truncating-store bias);
// c2, c3 quantized to fp16 (c3 loads as an fp16a SFPLOADI immediate seed,
// c2 lives in the 4th programmable const reg LREG11 aka vConstNeg1).
constexpr float ERF_FAST_C0 = 1.1297672986984253f;
constexpr float ERF_FAST_C1 = -0.34958192706108093f;
constexpr float ERF_FAST_C2 = 0.07464599609375f;
constexpr float ERF_FAST_C3 = -0.006679534912109375f;
// S = f32(sqrt(1.1283792257308960 * 1.0045 * 2^-149)), applied twice
constexpr float ERF_FAST_S = 3.985362527411278e-23f;

// LREG11 is sfpi's reserved -1.0f (CREG_IDX_NEG_1); this kernel repurposes it as a fourth programmable constant
// for the duration of the op (every SFPU op init restores -1.0f via _init_sfpu_config_reg()). sfpi >= 7.80
// deletes reads/writes through vConstNeg1, so it is accessed through a plain vCReg handle, the same class
// vConstFloatPrgm0 uses.
constexpr sfpi::impl_::LRegFile::vCReg<sfpi::vFloat> ERF_FAST_C2_REG(sfpi::CREG_IDX_NEG_1);

inline void _init_erf_bf16_fast_() {
    ERF_FAST_C2_REG = ERF_FAST_C2;  // LREG11 (programmable on Blackhole)
    sfpi::vConstFloatPrgm0 = ERF_FAST_C1;
    sfpi::vConstFloatPrgm1 = ERF_FAST_C0;
    sfpi::vConstFloatPrgm2 = ERF_FAST_S;
}

// One face = 8 dst vectors.
// Main path: |y| = xc * P(xc^2), xc = min(|x|, 2), P deg-3 minimax (relative
// error); coefficients scaled by 1.0045 so the truncating bf16 SFPSTORE acts
// as round-to-nearest.  Denormal inputs: the MAD unit flushes denormal
// *inputs* to zero, so the main path yields 0 for them, but for the largest
// bf16 denormals the golden is the smallest normal (0x0080..0x008F).
// Branchless fix: SFPCAST the raw float bits (sign-magnitude int) to float
// (no flush), scale by c0*2^-149 in two muls by S = sqrt(c0*2^-149), merge
// with max() on magnitudes:
//   - denormal lanes: fix = c0 * m * 2^-133 exactly; main = 0 -> max picks fix
//   - m <= 112: fix output is denormal -> flushed to 0 (golden flushes too)
//   - normal lanes: fix = (E+frac)*c0*2^-126 <= |y_main| always -> main wins
// Two dst vectors are processed per iteration with interleaved chains to hide
// the SFPU's ~2-cycle result latency (throughput is 1/cycle for independent
// ops).  All four f32 constants live in programmable const regs (LREG11-14).
// Exhaustive HW-faithful simulation: max 2 ULP, tolerates +-3e-4 perturbation.
inline void _calculate_erf_bf16_fast_() {
#pragma GCC unroll 4
    for (size_t i = 0; i < 8; i += 2) {
        sfpi::vFloat x0 = sfpi::dst_reg[i];
        sfpi::vFloat x1 = sfpi::dst_reg[i + 1];
        sfpi::vFloat a0 = sfpi::abs(x0);
        sfpi::vFloat a1 = sfpi::abs(x1);
        // signed denormal-fix path: cast raw bits, scale by c0*2^-149 (two muls)
        sfpi::vFloat f0 = sfpi::convert<sfpi::vFloat>(sfpi::as<sfpi::vSMag>(x0), sfpi::RoundMode::Nearest);
        sfpi::vFloat f1 = sfpi::convert<sfpi::vFloat>(sfpi::as<sfpi::vSMag>(x1), sfpi::RoundMode::Nearest);
        f0 = f0 * sfpi::vConstFloatPrgm2;
        f1 = f1 * sfpi::vConstFloatPrgm2;
        f0 = f0 * sfpi::vConstFloatPrgm2;
        f1 = f1 * sfpi::vConstFloatPrgm2;
        sfpi::vFloat b0 = 2.0f;  // SFPLOADI swap scratch
        sfpi::vFloat b1 = 2.0f;
        sfpi::vFloat c0v = sfpi::min(a0, b0);  // xc = min(|x|, 2)
        sfpi::vFloat c1v = sfpi::min(a1, b1);
        sfpi::vFloat t0 = c0v * c0v;
        sfpi::vFloat t1 = c1v * c1v;
        sfpi::vFloat p0 = ERF_FAST_C3;  // fp16a SFPLOADI Horner seed
        sfpi::vFloat p1 = ERF_FAST_C3;
        p0 = p0 * t0 + ERF_FAST_C2_REG;  // + c2 (LREG11)
        p1 = p1 * t1 + ERF_FAST_C2_REG;
        p0 = p0 * t0 + sfpi::vConstFloatPrgm0;  // + c1
        p1 = p1 * t1 + sfpi::vConstFloatPrgm0;
        p0 = p0 * t0 + sfpi::vConstFloatPrgm1;  // + c0
        p1 = p1 * t1 + sfpi::vConstFloatPrgm1;
        sfpi::vFloat y0 = c0v * p0;  // |y_main|
        sfpi::vFloat y1 = c1v * p1;
        sfpi::vFloat fa0 = sfpi::abs(f0);  // |fix| for magnitude merge
        sfpi::vFloat fa1 = sfpi::abs(f1);
        y0 = sfpi::max(y0, fa0);
        y1 = sfpi::max(y1, fa1);
        sfpi::dst_reg[i] = sfpi::copysgn(y0, f0);  // truncating bf16 store
        sfpi::dst_reg[i + 1] = sfpi::copysgn(y1, f1);
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_erf() {
    if constexpr (!APPROXIMATION_MODE && !is_fp32_dest_acc_en && ITERATIONS == 8) {
        _calculate_erf_bf16_fast_();
        return;
    }
    if constexpr (
        (!APPROXIMATION_MODE && !is_fp32_dest_acc_en) &&
        !(!APPROXIMATION_MODE && !is_fp32_dest_acc_en && ITERATIONS == 8)) {
        // ITERATIONS != 8: erf_init<false, false> cannot see ITERATIONS and has programmed the fast kernel's
        // LREG11-14 over the state this path relies on: sfpu_reciprocal_init's vConstFloatPrgm0 = 2.0f (read by
        // sfpu_reciprocal<false> inside piecewise_rational_eval) and the architectural -1.0f in LREG11 that
        // sfpi-compiled code assumes (SFPCONFIG imm mode writes the default, as in _init_sfpu_config_reg).
        sfpu_reciprocal_init<APPROXIMATION_MODE>();
        TTI_SFPCONFIG(0, 11, 1);
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

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
void erf_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
    // Fast bf16 path: programs LREG11-14 after sfpu_reciprocal_init so its constants win (see
    // _init_erf_bf16_fast_). The common prologue (SFPU config reg + ADDR_MOD_7) is run by the
    // llk_math_eltwise_unary_sfpu_init callback overload before this function is called.
    if constexpr (!APPROXIMATION_MODE && !is_fp32_dest_acc_en) {
        _init_erf_bf16_fast_();
    }
}

}  // namespace ckernel::sfpu
