// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>
#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_polyval.h"
#include "ckernel_sfpu_sigmoid.h"
#include "sfpu/ckernel_sfpu_load_config.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_expm1.h"
#include "ckernel_sfpu_trigonometry.h"
#include "cmath_common.h"

namespace ckernel::sfpu {

// tanh(x): t = 0.5*expm1(abs(2*x)); sgn(x) * t / (t + 1)
sfpi_inline sfpi::vFloat _sfpu_tanh_fp32_accurate_(sfpi::vFloat x) {
    sfpi::vFloat a, r, s, f, w, y, scale, bias0;
    sfpi::vFloat j, t, rcp, x0, x1, y0;
    sfpi::vInt i, e, x_exp;
    sfpi::vMag m;

    // Calculate j = x * (2 * log2(e)), interleaved with a = abs(2*x), and i = round(abs(j)), clamped to [0, 255].

    j = x * sfpi::vConstFloatPrgm0;  // j = x * 2 * log2(e)
    a = x + x;
    // i = round(abs(j)), clamped to [0, 255].
    m = sfpi::convert<sfpi::vUInt8>(j, sfpi::RoundMode::Nearest);
    i = m;
    j = sfpi::convert<sfpi::vFloat>(m, sfpi::RoundMode::Nearest);

    a = sfpi::setsgn(a, 0);
    f = j * sfpi::vConstFloatPrgm1 + a;  // f = a - j * ln(2)

    // expm1(f)
    r = 1.974105835e-04f;
    r = r * f + 1.393318176e-3f;
    r = r * f + 8.331298828e-3f;
    r = r * f + 4.166680202e-2f;
    s = f * f;  // hide SFPMAD latency
    r = r * f + sfpi::vConstFloatPrgm2;
    w = 0.5f;
    r = __builtin_rvtt_sfpmad(r.get(), f.get(), w.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);

    e = i + 126;
    r = r * s + f;
    scale = sfpi::setexp(sfpi::vFloat(0.0f), e);
    bias0 = scale - w;

    // If a=±inf, converts to a finite value, otherwise if a=±NaN, converts to ±inf or ±NaN.
    // This gives y = <finite value> * 0.0 + 1.0 = 1.0 for non-NaN x, otherwise y = NaN.
    a = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(a) - 1);
    x0 = r * scale + bias0;
    y = a * 0.0f + 1.0f;
    x1 = x0 + 1.0f;

    // `i` is round(abs(2*x/log(2))). For i >= 61, |x| is about 21 or larger,
    // so x0/(x0 + 1) is far within 0.5 ulp of 1.0f. Keep the preinitialized
    // saturated result; below that, refine the reciprocal estimate.
    v_if(i < 61) {
        // computes x0/x1 via reciprocal and residual correction
        rcp = sfpi::approx_recip(x1);
        t = -x1 * rcp + 1.0f;
        y = x;
        rcp = rcp * t + rcp;
        y0 = x0 * rcp;
        x_exp = sfpi::exexp(x, sfpi::ExponentMode::Biased);
        t = -x1 * y0 + x0;

        // For tiny inputs, tanh(x) rounds to x in fp32. `x_exp` is biased, so
        // 115 is 127 - 12; keep y=x for |x| < 2^-12 and use the corrected
        // ratio otherwise.
        v_if(x_exp >= 115) { y = t * rcp + y0; }
        v_endif;
    }
    v_endif;

    return sfpi::copysgn(y, x);
}

// Sollya coefficients. tanh_init has programmable CRegs for the top three only, so these three
// cost an SFPLOADI pair per use unless the caller keeps them in an LReg.
// val * (0.999004364013671875 + val * (3.0897438526153564453125e-2 + val * (-0.4890659749507904052734375 + val *
// (0.281917631626129150390625 + val * (-6.6649019718170166015625e-2 + val *
// (5.876733921468257904052734375e-3))))));
constexpr float TANH_POLY_C1 = 0.999004364013671875f;
constexpr float TANH_POLY_C2 = 3.0897438526153564453125e-2f;
constexpr float TANH_POLY_C3 = -0.4890659749507904052734375f;

sfpi_inline sfpi::vFloat _sfpu_tanh_polynomial_(sfpi::vFloat x) {
    // For negative numbers, we compute tanh(-x) = -tanh(x)
    sfpi::vFloat val = sfpi::abs(x);  // set positive

    sfpi::vFloat result = PolynomialEvaluator::eval(
        val,
        0.0f,
        TANH_POLY_C1,
        TANH_POLY_C2,
        TANH_POLY_C3,
        sfpi::vConstFloatPrgm2,
        sfpi::vConstFloatPrgm1,
        sfpi::vConstFloatPrgm0);

    // For larger x, the polynomial approximation may exceed 1.0.
    // Since tanh(x) is bounded by [-1, 1], we clamp output to 1.0.
    result = sfpi::min(result, 1.0f);

    result = sfpi::copysgn(result, x);  // restore sign (i.e. tanh(-x) = -tanh(x))

    return result;
}

// Two datums through the polynomial in lockstep, so each fills the other's SFPMAD stall slots.
// Only WH stalls; BH comes out even either way, so both arches run this shape. Only c1 can be
// hoisted on top of it: six vectors are already live for the data, and an eighth spills.
sfpi_inline void _sfpu_tanh_polynomial_x2_(
    sfpi::vFloat& y0, sfpi::vFloat& y1, sfpi::vFloat x0, sfpi::vFloat x1, sfpi::vFloat c1) {
    sfpi::vFloat a0 = sfpi::abs(x0);
    sfpi::vFloat a1 = sfpi::abs(x1);

    sfpi::vFloat r0 = sfpi::vConstFloatPrgm0;
    sfpi::vFloat r1 = sfpi::vConstFloatPrgm0;
    r0 = r0 * a0 + sfpi::vConstFloatPrgm1;
    r1 = r1 * a1 + sfpi::vConstFloatPrgm1;
    r0 = r0 * a0 + sfpi::vConstFloatPrgm2;
    r1 = r1 * a1 + sfpi::vConstFloatPrgm2;
    // One local each, else sfpi emits the SFPLOADI pair per MAD. Both die after their second use.
    sfpi::vFloat c3 = TANH_POLY_C3;
    r0 = r0 * a0 + c3;
    r1 = r1 * a1 + c3;
    sfpi::vFloat c2 = TANH_POLY_C2;
    r0 = r0 * a0 + c2;
    r1 = r1 * a1 + c2;
    r0 = r0 * a0 + c1;
    r1 = r1 * a1 + c1;
    r0 = r0 * a0;
    r1 = r1 * a1;

    y0 = sfpi::copysgn(sfpi::min(r0, 1.0f), x0);
    y1 = sfpi::copysgn(sfpi::min(r1, 1.0f), x1);
}

// vConstFloatPrgm0..2 = the three top Sollya coefficients read by _sfpu_tanh_polynomial_ / _sfpu_tanh_polynomial_x2_.
sfpi_inline void _load_tanh_polynomial_constants_() {
    sfpi::vConstFloatPrgm0 = 5.876733921468257904052734375e-3;
    sfpi::vConstFloatPrgm1 = -6.6649019718170166015625e-2;
    sfpi::vConstFloatPrgm2 = 0.281917631626129150390625;
}

#ifndef DISABLE_SFPLOADMACRO
// =====================================================================================================================
// Fast bf16 tanh for Blackhole. Origin: llk-bench (LLM-agent-written kernel, claude-fable-5, 2026-08),
// validated exhaustively over all 65,536 bf16 inputs vs the exact golden: max 2 ULP (gate <= 2).
// Measured 375.1 cycles/tile vs 776.1 for the previous kernel on p150b (tt-metal v0.76.0 baseline).
//
// State programmed by _init_tanh_bf16_fast_: LREG11..14 (config path) and LREG6; SFPLOADMACRO instruction
// templates 0..2, macro sequences 0..2 (SFPCONFIG dest 4..6) and the LoadMacro Misc register (dest 8). No replay
// slots. Needs ADDR_MOD_7 = {0,0,0} (re-asserted by the init). bf16 DEST only: never use for fp32 dest or on
// Wormhole. Processes one face (8 dst vectors at ADDR_MOD_7 offsets 0,2,...,14) per call.
//
// bf16 tanh via odd polynomial x*P(x^2), deg-11.
//
//   y = x*(1 + c2 t + c3 t^2 + c4 t^3 + c5 t^4),  t = x^2
//   y = min(y, 1.0); y = max(y, -1.0); truncating store to bf16.
//   Coefficients LP-fitted for this exact pipeline (c1 = 1.0 exactly, fused
//   SFPMAD, denormal flush, truncation); exhaustive sweep: max ULP 2.
//
// Two interleaved chains (A: L0/L2, B: L3/L4/L5) hide the 2-cycle SFPMAD
// latency. SFPLOADMACRO does triple duty:
//   - A-head macro: loads x and schedules t = x*x in place (x is reloaded
//     from Dst at the tail, so the head copy is dead after this).
//   - A/B tail macros: reload x, schedule y = r*x into the loaded reg, and
//     schedule the SFPSTORE of the clamped result back to the same address
//     (instruction-counted delay 5, i.e. after both regular SFPSWAP clamps).
// LOADMACROs are never issued back-to-back (hardware corrupts scheduled ops)
// and no regular MAD-unit instruction is issued on a cycle where a scheduled
// MAD executes.
//
// Constants: L6 = c2, L12 = c5, L13 = c4, L14 = c3, L11 = -1.0,
//            L9 = 0.0 (hw), L10 = 1.0 (hw, doubles as c1).
// =====================================================================================================================
inline void _init_tanh_bf16_fast_() {
    // Self-contained: re-assert the common SFPU state (config reg + ADDR_MOD_7 + RWC reset) as exp_init does,
    // then program this kernel's constants and SFPLOADMACRO templates/sequences.
    _init_sfpu_config_reg();
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    math::reset_counters(p_setrwc::SET_ABD_F);

    // Programmable constants (config path stages through LReg[0]).
    _sfpu_load_config32_(11, 0xBF80, 0x0000);  // L11 = -1.0
    _sfpu_load_config32_(12, 0x3A11, 0x57E6);  // L12 = c5
    _sfpu_load_config32_(13, 0xBC2C, 0xB110);  // L13 = c4
    _sfpu_load_config32_(14, 0x3D9E, 0xB72A);  // L14 = c3
    _sfpu_load_imm32_(6, 0xBE9C45A4);          // L6 = c2

    // InstructionTemplate[0]: tA = L0 * VB + 0 (VB routed to loaded reg L0,
    // dest overridden to L0 by the sequence -> in-place x -> x*x).
    TTI_SFPMAD(0, 0, p_sfpu::LCONST_0, 12, 0);
    // InstructionTemplate[1]: yA = rA(L2) * VB + 0 (VB routed to L0).
    TTI_SFPMAD(2, 0, p_sfpu::LCONST_0, 13, 0);
    // InstructionTemplate[2]: yB = rB(L5) * VB + 0 (VB routed to L3).
    TTI_SFPMAD(5, 3, p_sfpu::LCONST_0, 14, 0);

    // Sequence bytes: [2:0] code (3=store, 4..7=template0..3), [5:3] delay,
    // [6] dest=L16, [7] route loaded reg to VB; unused sub-units use
    // delay 7 (avoids the same-delay forget of pending scheduled ops).
    // Macro 0 (A-head): MAD 0x84 = route-VB | delay 0 | template 0.
    TTI_SFPLOADI(0, 10, 0x8438);
    TTI_SFPLOADI(0, 8, 0x3838);
    TTI_SFPCONFIG(0, 4, 0);
    // Macro 1 (A-tail): MAD 0x85 (delay 0, template 1); STORE 0x2B (delay 5).
    TTI_SFPLOADI(0, 10, 0x8538);
    TTI_SFPLOADI(0, 8, 0x2B38);
    TTI_SFPCONFIG(0, 5, 0);
    // Macro 2 (B-tail): MAD 0x8E (delay 1, template 2); STORE 0x2B (delay 5).
    TTI_SFPLOADI(0, 10, 0x8E38);
    TTI_SFPLOADI(0, 8, 0x2B38);
    TTI_SFPCONFIG(0, 6, 0);
    // Misc: [3:0] StoreMod0=0, [7:4] UsesLoadMod0ForStore (macros 1,2),
    // [11:8] UnitDelayKind = instruction-counted on all sub-units.
    TTI_SFPCONFIG(0xF60, 8, 1);
    TTI_SFPNOP;
}

// One pair of vectors (dest row offsets o1, o2, both even so the address
// LSB stolen as VDHi stays 0). lreg_ind = (macro << 2) | (VD & 3).
#define TANH_FAST_PAIR(o1, o2)                                    \
    TT_SFPLOADMACRO((0 << 2) | 0, 0, ADDR_MOD_7, o1); /* xA,tA */ \
    TTI_SFPLOAD(3, 0, ADDR_MOD_7, o2);                /* xB */    \
    TTI_SFPMAD(3, 3, p_sfpu::LCONST_0, 4, 0); /* tB = x*x */      \
    TTI_SFPMAD(0, 12, 13, 2, 0); /* rA = c5*t + c4 */             \
    TTI_SFPMAD(4, 12, 13, 5, 0);                                  \
    TTI_SFPMAD(2, 0, 14, 2, 0); /* rA = rA*t + c3 */              \
    TTI_SFPMAD(5, 4, 14, 5, 0);                                   \
    TTI_SFPMAD(2, 0, 6, 2, 0); /* rA = rA*t + c2 */               \
    TTI_SFPMAD(5, 4, 6, 5, 0);                                    \
    TTI_SFPMAD(2, 0, p_sfpu::LCONST_1, 2, 0); /* rA = rA*t + 1 */ \
    TTI_SFPMAD(5, 4, p_sfpu::LCONST_1, 5, 0);                     \
    TT_SFPLOADMACRO((1 << 2) | 0, 0, ADDR_MOD_7, o1); /* yA,st */ \
    TTI_SFPNOP;                                                   \
    TT_SFPLOADMACRO((2 << 2) | 3, 0, ADDR_MOD_7, o2); /* yB,st */ \
    TTI_SFPSWAP(0, p_sfpu::LCONST_1, 0, 1); /* yA=min(yA,1) */    \
    TTI_SFPSWAP(0, 11, 0, 9); /* yA = max(yA,-1) */               \
    TTI_SFPSWAP(0, p_sfpu::LCONST_1, 3, 1); /* yB=min(yB,1) */    \
    TTI_SFPSWAP(0, 11, 3, 9); /* yB = max(yB,-1) */

inline void _calculate_tanh_bf16_fast_() {
    TANH_FAST_PAIR(0, 2);
    TANH_FAST_PAIR(4, 6);
    TANH_FAST_PAIR(8, 10);
    TANH_FAST_PAIR(12, 14);
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

#undef TANH_FAST_PAIR
#endif  // !DISABLE_SFPLOADMACRO

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_tanh() {
#ifndef DISABLE_SFPLOADMACRO
    if constexpr (!APPROXIMATION_MODE && !is_fp32_dest_acc_en && ITERATIONS == 8) {
        _calculate_tanh_bf16_fast_();
        return;
    }
#endif
    if constexpr (APPROXIMATION_MODE) {
        // Slopes in LReg0/1/2 packed hi/lo, intercepts in LReg4/5/6 -- where WH and BH keep a
        // 6-entry SFPLUTFP32 table. gelu_appx uses the same six registers the same way.
        sfpi::vLut16ss s01 = l_reg[sfpi::LRegs::LReg0];
        sfpi::vLut16ss s23 = l_reg[sfpi::LRegs::LReg1];
        sfpi::vLut16ss s45 = l_reg[sfpi::LRegs::LReg2];
        sfpi::vLut16ii i01 = l_reg[sfpi::LRegs::LReg4];
        sfpi::vLut16ii i23 = l_reg[sfpi::LRegs::LReg5];
        sfpi::vLut16ii i45 = l_reg[sfpi::LRegs::LReg6];

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];
            val = sfpi::lut(val, s01, i01, s23, i23, s45, i45, sfpi::LutSign::Retain);
            sfpi::dst_reg[0] = val;

            sfpi::dst_reg++;
        }

        l_reg[sfpi::LRegs::LReg0] = s01;
        l_reg[sfpi::LRegs::LReg1] = s23;
        l_reg[sfpi::LRegs::LReg2] = s45;
        l_reg[sfpi::LRegs::LReg4] = i01;
        l_reg[sfpi::LRegs::LReg5] = i23;
        l_reg[sfpi::LRegs::LReg6] = i45;
    } else if constexpr (is_fp32_dest_acc_en) {  // APPROXIMATION_MODE is false
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];
            sfpi::vFloat result = _sfpu_tanh_fp32_accurate_(val);
            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
    } else {
#ifndef DISABLE_SFPLOADMACRO
        if constexpr (ITERATIONS != 8) {
            // tanh_init<false, false> cannot see ITERATIONS and has programmed the fast kernel's LREG12..14 on
            // top of the polynomial constants this path reads from vConstFloatPrgm0..2; re-seed them.
            _load_tanh_polynomial_constants_();
        }
#endif
        sfpi::vFloat c1 = TANH_POLY_C1;  // inline it and every datum pays an SFPLOADI pair

        // Walk dst_reg rather than index by d: a uniform body is what the replay buffer records
        // once, and a runtime index makes sfpi build each SFPLOAD/SFPSTORE in scalar registers.
#pragma GCC unroll 4
        for (int d = 0; d < ITERATIONS / 2; d++) {
            sfpi::vFloat r0, r1;
            _sfpu_tanh_polynomial_x2_(r0, r1, sfpi::dst_reg[0], sfpi::dst_reg[1], c1);
            // Round into a vFloat; storing the vFloat16b expression pins SFPSTORE to FP16B.
            r0 = sfpi::convert<sfpi::vFloat16b>(r0, sfpi::RoundMode::Nearest);
            r1 = sfpi::convert<sfpi::vFloat16b>(r1, sfpi::RoundMode::Nearest);

            sfpi::dst_reg[0] = r0;
            sfpi::dst_reg[1] = r1;
            sfpi::dst_reg += 2;
        }

        if constexpr (ITERATIONS % 2 != 0) {
            sfpi::vFloat result = _sfpu_tanh_polynomial_(sfpi::dst_reg[0]);
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);

            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
    }
}

// Programs the vConstFloatPrgm0..2 / LUT state that every tanh-polynomial consumer reads: calculate_tanh's
// fallback paths and the _sfpu_tanh_polynomial_ / _sfpu_tanh_fp32_accurate_ users (softcap, situ_glu,
// logit_softcap, gelu_tanh). Deliberately free of the bf16 fast-path init below, whose LREG12..14 values would
// clobber these constants -- consumers other than calculate_tanh must call this, not tanh_init.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void tanh_init_constants() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    if constexpr (APPROXIMATION_MODE) {
        // 6-entry SFPLUTFP32 FP16 table, TABLE1 breakpoints |x| = 0.5, 1.0, 1.5, 2.0, 3.0.
        // SGN_RETAIN, so the result is sign(x) * (A*|x| + B) and the kernel stays odd.
        // Reached only by callers passing fast_and_approx; gelu, softcap and situ_glu call
        // tanh_init with APPROXIMATION_MODE=false and never load these registers.
        //
        // Fitted to minimise max bfloat16 ULP error, not max absolute error. To retune, keep:
        //  - segment 0's intercept at 0, else SGN_RETAIN puts a jump across the origin;
        //  - the last segment at exactly (0, 1.0), so *finite* inputs saturate to 1.0. It says
        //    nothing about the infinities: the hardware evaluates A*|x| + B, so 0 * inf + 1 is
        //    NaN rather than 1.0. That predates this table; test_tanh_specials records it;
        //  - every segment <= 1.0 over its own range -- unlike the polynomial path below, this
        //    one has no min(result, 1.0f) to fall back on;
        //  - no step down where two segments meet. A bfloat16 sweep tolerates a step under the
        //    ~2e-3 ulp there, but this kernel has no convert<vFloat16b> and also serves the
        //    fp32-dest path, where a 1.2e-4 dip is ~2048 fp32 ulp of non-monotonicity. So the
        //    intercept of segment 1 and the slope of segment 3 are held one fp16 ulp below
        //    their minimax values, which lands the joins at |x| = 0.5, 1.0 and 2.0 exactly and
        //    leaves a single upward step of +1.2e-4 at |x| = 1.5. It is free: the bfloat16
        //    sweep is identical either way -- 10.00 max ULP, 0.018352 max absolute error, same
        //    percentiles -- while fp32 max absolute error improves 0.018962 -> 0.018840.
        //
        // test_tanh_lut_consistency.py checks all four against the header, and holds the two
        // arch copies of this table together.
        sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vLut16ss(0.96191406f, 0.57617188f);
        sfpi::l_reg[sfpi::LRegs::LReg4] = sfpi::vLut16ii(0.0f, 0.192871094f);

        sfpi::l_reg[sfpi::LRegs::LReg1] = sfpi::vLut16ss(0.28710938f, 0.0964355469f);
        sfpi::l_reg[sfpi::LRegs::LReg5] = sfpi::vLut16ii(0.48193359f, 0.76806641f);

        // 0.0390625 == 1.25 * 2^-5, fp16-exact, and chosen so A*3 + B is exactly 1.0: the
        // minimax slope 0.039123535 crosses 1.0 at |x| = 2.99532 and peaks at 1.000183.
        sfpi::l_reg[sfpi::LRegs::LReg2] = sfpi::vLut16ss(0.0390625f, 0.0f);
        sfpi::l_reg[sfpi::LRegs::LReg6] = sfpi::vLut16ii(0.8828125f, 1.0f);
    } else {
        if constexpr (is_fp32_dest_acc_en) {
            sfpi::vConstFloatPrgm0 = 2.0f * 1.442695f;      // 2 * log2(e) == 2 / ln(2)
            sfpi::vConstFloatPrgm1 = -0.6931471805599453f;  // ln(2)
            sfpi::vConstFloatPrgm2 = 1.666667163e-1f;       // c1
        } else {
            // Polynomial approximation
            // Store some polynomial coefficients in programmable registers
            _load_tanh_polynomial_constants_();
        }
    }
}

// Init for calculate_tanh (tanh_tile_init / tanh_tile_init_pack): the shared constants, then -- for the bf16
// non-approx configuration calculate_tanh<false, false, 8> serves with the fast kernel -- the fast kernel's own
// state on top, so its LREG11..14 values win. calculate_tanh<false, false, ITERATIONS != 8> re-seeds the
// polynomial constants itself.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void tanh_init() {
    tanh_init_constants<APPROXIMATION_MODE, is_fp32_dest_acc_en>();
#ifndef DISABLE_SFPLOADMACRO
    if constexpr (!APPROXIMATION_MODE && !is_fp32_dest_acc_en) {
        _init_tanh_bf16_fast_();
    }
#endif
}

}  // namespace ckernel::sfpu
