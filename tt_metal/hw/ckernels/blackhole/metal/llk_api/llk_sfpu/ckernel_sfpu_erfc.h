// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"

#include "ckernel_sfpu_piecewise_rational.h"

namespace ckernel::sfpu {

// ======================================================================
// LUT-based erfc via piecewise rational P(x)/Q(x)
//
// Uses abs(x) symmetry: erfc(-x) = 2 - erfc(x)
// Fit on [0, 5.0] only, 2 segments with n4/d5 rational per segment.
// BF16 MaxULP=118 (was 128 with 3-seg n4/d4 on [-5,5])
// FP32 MaxULP≈9M  (was 1.47B)
// 18 FMAs          (was 24)
// ======================================================================

constexpr uint32_t ERFC_NUM_DEGREE = 4;
constexpr uint32_t ERFC_DEN_DEGREE = 5;
constexpr uint32_t ERFC_NUM_SEGMENTS = 2;
constexpr uint32_t ERFC_LUT_SIZE = 25;
constexpr std::array<float, ERFC_LUT_SIZE> ERFC_LUT = {{// Breakpoints
                                             0.0000000000e+00f,
                                             2.5000000000e+00f,
                                             5.0000000000e+00f,
                                             // Segment 0 [0, 2.5]: numerator (degree 4)
                                             1.0000233650e+00f,
                                             -1.3375675678e+00f,
                                             6.8185544014e-01f,
                                             -1.5691982210e-01f,
                                             1.3746744953e-02f,
                                             // Segment 0 [0, 2.5]: denominator (degree 5)
                                             1.0000000000e+00f,
                                             -2.0801517367e-01f,
                                             4.3667086959e-01f,
                                             -3.4568668343e-03f,
                                             2.5104774162e-02f,
                                             2.8375532478e-02f,
                                             // Segment 1 [2.5, 5.0]: numerator (degree 4)
                                             -2.5655237550e-05f,
                                             2.1275576728e-05f,
                                             -6.6162156145e-06f,
                                             9.1439767402e-07f,
                                             -4.7387182178e-08f,
                                             // Segment 1 [2.5, 5.0]: denominator (degree 5)
                                             1.0000000000e+00f,
                                             -1.6457208991e-01f,
                                             -2.0572184026e-01f,
                                             -1.3888636231e-01f,
                                             1.2677097321e-01f,
                                             -2.1375391632e-02f}};

// ======================================================================
// Fast bf16 erfc for Blackhole. Origin: llk-bench (LLM-agent-written kernel, claude-fable-5, 2026-08),
// validated exhaustively over all 65,536 bf16 inputs vs the exact golden: max 1 ULP (gate <= 2).
// Measured 879.1 cycles/tile vs 2631.1 for the previous kernel on p150b (tt-metal v0.76.0 baseline).
//
// erfc via u*exp2(Q(u) - a^2*log2(e)) with u = 1/(a+2), Moroz-style bit-trick exp2, one Newton step on
// SFPARECIP.
//
// erfc(a) = u * 2^( Q(u) - a^2*log2(e) + B - 127 ),  a = |x| clamped to 9.25
//   Q(u)  = C3 u^3 + C2 u^2 + C1 u          (fit of log2(erfc(a)(a+2)e^{a^2}))
//   B     = fit c0 + 127 + log2(s_exp) + centering for the truncating store
// erfc(-a) = 2 - erfc(a).  erfc(+inf) = +0, erfc(-inf) = 2.
//
// Hand-scheduled TTI, two rows (A/B) interleaved per block. SFPLOADMACRO is
// used twice per row (behavior verified on silicon):
//   - front loads run template0 (SETSGN imm-neg -> na = -|x|) on the Simple
//     unit in the load shadow;
//   - reload loads schedule the Store unit at delay 7, so the final result
//     (written into the loaded reg by the out MAD at +6) stores itself.
// SFPARECIP runs on the negative argument (sign-preserving); SFPMAD
// NEGATE_VA/NEGATE_VC fix up the Newton step. CREG11 (normally -1) holds the
// tail clamp -9.25; SFPSWAP reads it and its CREG write is dropped.
//
// State programmed by init: LREG11 (vConstNeg1 := -9.25, i.e. NOT -1 while this op is active), LREG12 =
// -log2(e), LREG13 = B, LREG14 = E1; SFPLOADMACRO InstructionTemplate[0], macro sequences 0 and 1, and
// LoadMacroConfig misc (0x120). No replay slots. Hardware constants L9 = 0, L10 = 1 are read as-is.
// bf16 DEST only. Selected by calculate_erfc / erfc_init when !is_fp32_dest_acc_en (&& ITERATIONS == 8),
// unless DISABLE_SFPLOADMACRO is defined.
// ======================================================================

inline void _init_erfc_bf16_fast_() {
    sfpi::vConstFloatPrgm0 = -1.4426950408889634f;  // L12 = -log2(e)
    sfpi::vConstFloatPrgm1 = 126.18802642822266f;   // L13 = B (0x42fc6045)
    sfpi::vConstFloatPrgm2 = 7.857097e-08f;         // L14 = E1 (0x33a8bad9)
    // L11 (vConstNeg1) := -9.25f -- this kernel never uses -1.
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_FLOATB, 0xC114);
    TTI_SFPCONFIG(0, 0xB, 0);

    // InstructionTemplate[0]: dest/srcC are VD-substituted: VD = -|VD|
    TTI_SFPSETSGN(1, 0, 12, 1);
    // Macro 0: Simple unit runs template0 at delay 0, result to VD.
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, 0x04);    // simple=(0<<3)|(4+0), mad=0
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, 0x0000);  // round=0, store=0
    TTI_SFPCONFIG(0, 4, 0);
    // Macro 1: Store unit stores VD at delay 7 (elapsed instructions).
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, 0x0000);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, ((0x00 | (7 << 3) | 3) << 8) | 0);
    TTI_SFPCONFIG(0, 5, 0);
    // Load Macro Control: Simple unit WaitForElapsedInstructions (bit 8),
    // UsesLoadMod0ForStore for macro 1 (bit 5).
    TTI_SFPCONFIG(0x120, 8, 1);
}

// One interleaved pair of dst rows at offsets OA and OB = OA + 2.
template <int OA>
inline __attribute__((always_inline)) void _erfc_bf16_fast_pair_() {
    constexpr int OB = OA + 2;
    constexpr unsigned MOD0 = 0;          // InstrModLoadStore::DEFAULT
    constexpr unsigned NEG_TWO = 0xC000;  // -2.0f   (bf16 imm)
    constexpr unsigned E2_BF16 = 0x27AC;  // 4.773959e-15f
    constexpr unsigned C3_FP16 = 0xC406;  // -4.0234375f  (fp16a imm)
    constexpr unsigned C2_FP16 = 0x4403;  // 4.01171875f
    constexpr unsigned C1_FP16 = 0x4142;  // 2.62890625f

    // ---- front: na = max(-|x|, -9.25); s' = B - a^2*log2(e); u ~ 1/(a+2)
    TTI_SFPLOADMACRO((0 << 2) | 0, MOD0, ADDR_MOD_7, OA);   // L0 = xA -> naA
    TTI_SFPLOADMACRO((0 << 2) | 2, MOD0, ADDR_MOD_7, OB);   // L2 = xB -> naB
    TTI_SFPLOADI(3, sfpi::SFPLOADI_MOD0_FLOATA, C2_FP16);   // L3 = C2 (macro shadow)
    TTI_SFPLOADI(6, sfpi::SFPLOADI_MOD0_FLOATA, C1_FP16);   // L6 = C1 (macro shadow)
    TTI_SFPSWAP(0, 0, 11, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);  // L0 = max(naA,-9.25)
    TTI_SFPSWAP(0, 2, 11, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);  // L2 = max(naB,-9.25)
    TTI_SFPMAD(0, 12, 9, 1, 0);                             // L1 = vvA
    TTI_SFPMAD(2, 12, 9, 5, 0);                             // L5 = vvB
    TTI_SFPMAD(1, 0, 13, 1, 0);                             // L1 = s'A
    TTI_SFPMAD(5, 2, 13, 5, 0);                             // L5 = s'B
    TTI_SFPADDI(NEG_TWO, 0, 0);                             // L0 = ndA
    TTI_SFPADDI(NEG_TWO, 2, 0);                             // L2 = ndB
    TTI_SFPARECIP(0, 0, 7, sfpi::SFPARECIP_MOD1_RECIP);     // L7 = y0A (negative)
    TTI_SFPARECIP(0, 2, 4, sfpi::SFPARECIP_MOD1_RECIP);     // L4 = y0B
    TTI_SFPMAD(0, 7, 10, 0, 1);                             // L0 = eA = 1 - ndA*y0A
    TTI_SFPMAD(2, 4, 10, 2, 1);                             // L2 = eB
    TTI_SFPMAD(7, 0, 7, 0, 3);                              // L0 = uA = -y0A*eA - y0A
    TTI_SFPMAD(4, 2, 4, 4, 3);                              // L4 = uB
    // ---- Horner: xl = ((C3*u + C2)*u + C1)*u + s'
    TTI_SFPLOADI(2, sfpi::SFPLOADI_MOD0_FLOATA, C3_FP16);  // L2 = C3
    TTI_SFPMAD(2, 0, 3, 7, 0);                             // L7 = qA
    TTI_SFPMAD(2, 4, 3, 2, 0);                             // L2 = qB
    TTI_SFPMAD(7, 0, 6, 7, 0);                             // qA = qA*uA + C1
    TTI_SFPMAD(2, 4, 6, 2, 0);                             // qB
    TTI_SFPMAD(7, 0, 1, 1, 0);                             // L1 = xlA
    TTI_SFPMAD(2, 4, 5, 5, 0);                             // L5 = xlB
    // ---- exp2 integer part
    TTI_SFPEXEXP(0, 1, 2, sfpi::SFPEXEXP_MOD1_DEBIAS);  // L2 = exA
    TTI_SFPEXEXP(0, 5, 6, sfpi::SFPEXEXP_MOD1_DEBIAS);  // L6 = exB
    TTI_SFPEXMAN(0, 1, 3, sfpi::SFPEXMAN_MOD1_PAD8);    // L3 = manA
    TTI_SFPEXMAN(0, 5, 7, sfpi::SFPEXMAN_MOD1_PAD8);    // L7 = manB
    TTI_SFPSHFT(0, 2, 3, 0);                            // L3 = ziA
    TTI_SFPSHFT(0, 6, 7, 0);                            // L7 = ziB
    TTI_SFPEXMAN(0, 3, 2, sfpi::SFPEXMAN_MOD1_PAD9);    // L2 = frA
    TTI_SFPEXMAN(0, 7, 6, sfpi::SFPEXMAN_MOD1_PAD9);    // L6 = frB
    TTI_SFPCAST(2, 2, 0);                               // L2 = fA
    TTI_SFPLOADI(1, sfpi::SFPLOADI_MOD0_FLOATB, E2_BF16);  // L1 = E2
    TTI_SFPCAST(6, 6, 0);                                  // L6 = fB
    // ---- exp2 fraction poly: g = (E2*f + E1)*f + 1
    TTI_SFPMAD(2, 1, 14, 5, 0);  // L5 = gA
    TTI_SFPMAD(6, 1, 14, 1, 0);  // L1 = gB
    TTI_SFPMAD(5, 2, 10, 5, 0);  // gA = gA*fA + 1
    TTI_SFPMAD(1, 6, 10, 1, 0);  // gB = gB*fB + 1
    TTI_SFPSETEXP(0, 5, 3, 2);   // L3 = yA
    TTI_SFPSETEXP(0, 1, 7, 2);   // L7 = yB
    TTI_SFPMAD(0, 3, 9, 0, 0);   // L0 = rA
    TTI_SFPMAD(4, 7, 9, 4, 0);   // L4 = rB
    // ---- sign restore + self-storing reload macros
    TTI_SFPLOADMACRO((1 << 2) | 1, MOD0, ADDR_MOD_7, OA);  // L1 = xA; store L1 @+7
    TTI_SFPLOADMACRO((1 << 2) | 2, MOD0, ADDR_MOD_7, OB);  // L2 = xB; store L2 @+7
    TTI_SFPSETSGN(0, 10, 1, 0);                            // L1 = qsA
    TTI_SFPSETSGN(0, 10, 2, 0);                            // L2 = qsB
    TTI_SFPMAD(1, 10, 10, 6, 1);                           // L6 = wA = 1 - qsA
    TTI_SFPMAD(2, 10, 10, 7, 1);                           // L7 = wB
    TTI_SFPMAD(1, 0, 6, 1, 0);                             // L1 = outA (stored @+7)
    TTI_SFPMAD(2, 4, 7, 2, 0);                             // L2 = outB (stored @+7)
}

// One face = 8 dst vectors (four interleaved pairs).
inline void _calculate_erfc_bf16_fast_() {
    _erfc_bf16_fast_pair_<0>();
    _erfc_bf16_fast_pair_<4>();
    _erfc_bf16_fast_pair_<8>();
    _erfc_bf16_fast_pair_<12>();
}

template <bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_erfc() {
#ifndef DISABLE_SFPLOADMACRO
    if constexpr (!is_fp32_dest_acc_en && ITERATIONS == 8) {
        _calculate_erfc_bf16_fast_();
        return;
    }
    if constexpr (!is_fp32_dest_acc_en && !(!is_fp32_dest_acc_en && ITERATIONS == 8)) {
        // ITERATIONS != 8: erfc_init<.., false> cannot see ITERATIONS and has programmed the fast kernel's
        // LREG11 (-9.25f) over the architectural -1.0f that sfpi-compiled code assumes; restore it (SFPCONFIG
        // imm mode writes the default, as in _init_sfpu_config_reg). Its LREG12-14 values are harmless here:
        // sfpu_reciprocal_init<true>() programs nothing and sfpu_reciprocal<true> (inside
        // piecewise_rational_eval) reads no programmable constant.
        sfpu_reciprocal_init<true>();
        TTI_SFPCONFIG(0, 11, 1);
    }
#endif
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        // Clamp |x| to 5.0 before evaluation (avoids extrapolation, saves one branch)
        sfpi::vFloat ax = sfpi::min(sfpi::abs(x), 5.0f);
        sfpi::vFloat r =
            piecewise_rational_eval<ERFC_NUM_DEGREE, ERFC_DEN_DEGREE, ERFC_NUM_SEGMENTS, ERFC_LUT_SIZE, false, true>(
                ERFC_LUT, ax);
        // erfc(-x) = 2 - erfc(x)
        v_if(x < 0.0f) { r = 2.0f - r; }
        v_endif;
        sfpi::dst_reg[0] = r;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
void erfc_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<true>();
#ifndef DISABLE_SFPLOADMACRO
    // Fast bf16 path: programs LREG11-14 and the SFPLOADMACRO state after sfpu_reciprocal_init so its
    // values win (see _init_erfc_bf16_fast_). The common prologue (SFPU config reg + ADDR_MOD_7) is run
    // by the llk_math_eltwise_unary_sfpu_init callback overload before this function is called.
    if constexpr (!is_fp32_dest_acc_en) {
        _init_erfc_bf16_fast_();
    }
#endif
}

}  // namespace ckernel::sfpu
