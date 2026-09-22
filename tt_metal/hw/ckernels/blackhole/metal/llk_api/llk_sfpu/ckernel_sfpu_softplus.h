// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpu/ckernel_sfpu_load_config.h"
#include "ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_polyval.h"

namespace ckernel::sfpu {

// ======================================================================
// Softplus via abs(x) symmetry + residual function
//
// Uses the identity: softplus(-x) = softplus(x) - x
// Defining f(a) = ln(1 + exp(-a)) for a >= 0:
//   softplus(t) = t + f(t)   for t >= 0
//   softplus(t) = f(-t)      for t < 0
//
// FP32: degree-8 polynomial for f(a) on [0, 5] + inline exp + 3-term Taylor tail
// BF16: degree-6 polynomial (bf16-accurate, <0.28 ULP) + tail clamped to 0
//       (residual < exp(-5) = 0.0067 for a > 5, below bf16 rounding vs the t>0 term,
//        so the expensive exp tail is unnecessary at bf16 precision)
// ======================================================================

constexpr float SOFTPLUS_POLY_BOUNDARY = 5.0f;

// FP32 residual polynomial: f(a) = ln(1+exp(-a)) on [0, 5], degree 8
constexpr float SOFTPLUS_POLY_C0 = 6.9310557842e-01f;
constexpr float SOFTPLUS_POLY_C1 = -4.9926245213e-01f;
constexpr float SOFTPLUS_POLY_C2 = 1.2186349183e-01f;
constexpr float SOFTPLUS_POLY_C3 = 5.6753782555e-03f;
constexpr float SOFTPLUS_POLY_C4 = -1.0528374463e-02f;
constexpr float SOFTPLUS_POLY_C5 = 2.7290175203e-03f;
constexpr float SOFTPLUS_POLY_C6 = -3.4358495031e-04f;
constexpr float SOFTPLUS_POLY_C7 = 2.1285692128e-05f;
constexpr float SOFTPLUS_POLY_C8 = -4.8245715334e-07f;

// BF16 residual polynomial: f(a) = ln(1+exp(-a)) on [0, 5], degree 6
// (ULP-weighted minimax fit; max error < 0.28 bf16 ULP over the domain)
constexpr float SOFTPLUS_BF16_POLY_C0 = 6.9423984729e-01f;
constexpr float SOFTPLUS_BF16_POLY_C1 = -5.0932420424e-01f;
constexpr float SOFTPLUS_BF16_POLY_C2 = 1.4279095486e-01f;
constexpr float SOFTPLUS_BF16_POLY_C3 = -1.3000584069e-02f;
constexpr float SOFTPLUS_BF16_POLY_C4 = -1.8627923291e-03f;
constexpr float SOFTPLUS_BF16_POLY_C5 = 5.0152968088e-04f;
constexpr float SOFTPLUS_BF16_POLY_C6 = -3.1273466851e-05f;

// ======================================================================
// Lightweight inline exp(x) for negative x (tail region).
// Adapted from gelu's x_times_exp_negative_tail (ckernel_sfpu_gelu.h).
// Uses Cody-Waite range reduction + Taylor polynomial.
// BF16: degree 5 (~15 ops), FP32: degree 7 (~19 ops).
// ======================================================================
sfpi_inline sfpi::vFloat softplus_exp_negative(sfpi::vFloat x) {
    constexpr float INV_LN2 = 1.4426950408889634f;
    constexpr float LN2_HI = -0.6931152343750000f;
    constexpr float LN2_LO = -3.19461832987e-05f;

    // Range reduction: x = k*ln(2) + r
    sfpi::vFloat z = x * INV_LN2;
    sfpi::vInt k_int;
    sfpi::vFloat k = _sfpu_round_to_nearest_int32_(z, k_int);

    // Cody-Waite: r = x - k*ln(2) in extended precision
    sfpi::vFloat r = k * LN2_HI + x;
    r = k * LN2_LO + r;

    // exp(r) via Taylor polynomial, |r| < 0.5
#ifdef INP_FLOAT32
    // FP32: degree 7 for < 1 ULP
    sfpi::vFloat poly = PolynomialEvaluator::eval(
        r, 1.0f, 1.0f, 0.5f, 0.166666667f, 0.0416666667f, 0.00833333333f, 0.00138888889f, 0.000198412698f);
#else
    // BF16: degree 5 sufficient
    sfpi::vFloat poly = PolynomialEvaluator::eval(r, 1.0f, 1.0f, 0.5f, 0.166666667f, 0.0416666667f, 0.00833333333f);
#endif

    // Scale by 2^k via exponent manipulation
    sfpi::vInt p_exp = sfpi::exexp(poly, sfpi::ExponentMode::Biased);
    sfpi::vInt new_exp = p_exp + k_int;

    // FTZ: if exponent underflows, result is 0
    sfpi::vFloat result = 0.0f;
    v_if(new_exp > 0) { result = sfpi::setexp(poly, new_exp); }
    v_endif;

    return result;
}

#ifndef DISABLE_SFPLOADMACRO
// =====================================================================================================================
// Fast bf16 softplus for Blackhole (beta = 1, threshold = 20 only). Origin: llk-bench (LLM-agent-written kernel,
// claude-fable-5, 2026-08), validated exhaustively over all 65,536 bf16 inputs vs the exact golden with
// beta = 1.0, threshold = 20.0: max 1 ULP (gate <= 2).
// Measured 559.1 cycles/tile vs 1519.1 for the previous kernel on p150b (tt-metal v0.76.0 baseline).
//
// State programmed by _init_softplus_bf16_fast_: LREG12..14 (vConstFloatPrgm0..2) and LREG11 = -1.0 (config
// path); SFPLOADMACRO macro instruction 0 (SFPCONFIG dest 0) plus backdoor templates 1/2, macro sequences 0/1
// (SFPCONFIG dest 4/5) and a reset of the LoadMacro Misc register (dest 8). No replay slots. Needs ADDR_MOD_7 =
// {0,0,0} (re-asserted by the init). None of this state is read by the generic calculate_softplus_body, so the
// runtime fallback for other beta/threshold values is unaffected by it. bf16 DEST only: never use for fp32 dest
// or on Wormhole. Processes one face (8 dst vectors at ADDR_MOD_7 offsets 0,2,...,14) per call.
//
// softplus, hand-scheduled TTI, 2-way interleaved, SFPLOADMACRO-fused tail.
//
// softplus(x) = relu(x) + ln(1 + E),  E = exp(-|x|), branchless:
//   w  = max(-|x| * log2(e) + 127, 0)    (0 iff |x| >= 88.03, incl +-inf)
//   k  = uint8_sat(trunc(w))             (SFP_STOCH_RND FP32->UINT8, rnd-zero;
//                                         converts by magnitude -- w >= 0 here)
//   r  = w - k in [0, 1)
//   p  = 1 + c1*r + c2*r^2 ~= 2^r        (in [1, 2), fitted minimax)
//   E  = setexp(p, k)                    = p * 2^(k-127) = exp(-|x|)
//   y  = relu(x) + E * (1 + q1 E + q2 E^2)  [q ~= ln(1+E)/E, tuned exhaustively]
// Underflow lands on k = 0 -> E subnormal/zero -> flushes at pack/compare.
// Exhaustive bit-exact simulation + on-card sweep: max ULP 1 (gate is 2).
// bf16 output written by truncating store (error budget verified).
//
// Two dst vectors per unrolled block, chains A (L1,L2,+L3) / B (L5,L6,+L7)
// interleaved to hide SFPU latency. The tail (relu + final MAD + store) is
// fused into one SFPLOADMACRO per element: it re-loads x from dst (still
// intact), SWAPs it against 0 (relu), runs y = q*E + relu on the MAD unit,
// and stores y to the same dst address.
//
// Empirical macro-dispatch rules honored here:
//  - the slot after a LOADMACRO must be another LOADMACRO or SFPNOP
//  - micro-ops of concurrently in-flight macros must land on disjoint cycles
//  - explicit compute ops must not co-issue on a macro's SIMPLE/MAD cycle
//    (loads are OK on MAD/STORE cycles; compute is OK on STORE cycles)
//  - SFPSWAP occupies the SIMPLE unit for 2 cycles (not pipelined)
// B's macro: LM at t   -> SWAP@t..t+1, MAD@t+2, STORE@t+4  (lreg L0)
// A's macro: LM at t+1 -> SWAP@t+3..4, MAD@t+5, STORE@t+7  (lreg L3)
// Shadow slots: NOP, NOP, next loads (L1,L5), next SETSGNs.
//
// Macro instructions: mux4 = SWAP(0, sub, L9, min_max)  [shared relu]
//                     mux5 = MAD(L1, L2, sub -> sub)    [chain A tail]
//                     mux6 = MAD(L5, L6, sub -> sub)    [chain B tail]
//
// Const regs: L11 = -1.0, L12 = log2e, L13 = 127.0, L14 = c1 = 0.6660174
// c2/q2/q1 via fp16a SFPLOADI shared by the pair (scratch L3/L7).
// =====================================================================================================================
inline void _init_softplus_bf16_fast_() {
    // Self-contained: re-assert the common SFPU state (config reg + ADDR_MOD_7 + RWC reset) as exp_init does,
    // then program this kernel's constants and SFPLOADMACRO instructions/sequences.
    _init_sfpu_config_reg();
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    math::reset_counters(p_setrwc::SET_ABD_F);

    sfpi::vConstFloatPrgm0 = 1.4426950408889634f;  // log2(e)  -> L12
    sfpi::vConstFloatPrgm1 = 127.0f;               // exp bias -> L13
    sfpi::vConstFloatPrgm2 = 0.6660173829734749f;  // c1       -> L14
    // L11 (CREG_PRGM0) = -1.0f (its conventional value; set explicitly)
    TTI_SFPLOADI(0, 0xA, 0x0000);
    TTI_SFPLOADI(0, 0x8, 0xBF80);
    TTI_SFPCONFIG(0, 11, 0);

    // --- LoadMacro setup (patterned after ckernel_sfpu_exp.h) ---
    // Macro instruction 0 (mux4): SFPSWAP(0, sub, L9, VEC_MIN_MAX) -> relu
    TTI_SFPLOADI(0, 0xA, 0x0091);
    TTI_SFPLOADI(0, 0x8, 0x9200);
    TTI_SFPCONFIG(0, 0, 0);
    TTI_SFPNOP;
    // Macro instruction 1 (mux5), backdoor: y = L1*L2 + sub -> sub
    TTI_SFPMAD(1, 2, 0, 13, 0);
    // Macro instruction 2 (mux6), backdoor: y = L5*L6 + sub -> sub
    TTI_SFPMAD(5, 6, 0, 14, 0);
    // Sequence 0 (config dest 4), chain B: SIMPLE=mux4@0, MAD=mux6@2, STORE@4
    TTI_SFPLOADI(0, 0xA, 0x1604);
    TTI_SFPLOADI(0, 0x8, 0x2300);
    TTI_SFPCONFIG(0, 4, 0);
    // Sequence 1 (config dest 5), chain A: SIMPLE=mux4@0, MAD=mux5@2, STORE@4
    TTI_SFPLOADI(0, 0xA, 0x1504);
    TTI_SFPLOADI(0, 0x8, 0x2300);
    TTI_SFPCONFIG(0, 5, 0);
    // Reset LoadMacroConfig[Lane].Misc (as production exp_init does)
    TTI_SFPCONFIG(0, 8, 1);
}

// Pair body; x_A/x_B already negated-abs in L1/L5 (loaded + setsgn'd by the
// previous pair's tail or by the prologue). Tail prefetches next pair's x.
#define SOFTPLUS_FAST_PAIR(offA, offB, offNA, offNB)                 \
    TTI_SFPMAD(1, 12, 13, 1, 0);         /* A: w = n*L + 127     */  \
    TTI_SFPMAD(5, 12, 13, 5, 0);         /* B: w                 */  \
    TTI_SFPSWAP(0, 1, 9, 1);             /* A: w = max(w,0)      */  \
    TTI_SFPSWAP(0, 5, 9, 1);             /* B: w = max(w,0)      */  \
    TTI_SFP_STOCH_RND(2, 0, 0, 1, 2, 2); /* A: k = sat_u8_rz(w)  */  \
    TTI_SFP_STOCH_RND(2, 0, 0, 5, 6, 2); /* B: k                 */  \
    TTI_SFPCAST(2, 3, 0);                /* A: kf                */  \
    TTI_SFPCAST(6, 0, 0);                /* B: kf -> L0          */  \
    TTI_SFPMAD(3, 11, 1, 1, 0);          /* A: r = w - kf        */  \
    TTI_SFPMAD(0, 11, 5, 5, 0);          /* B: r                 */  \
    TTI_SFPMAD(7, 1, 14, 3, 0);          /* A: p1 = c2*r + c1    */  \
    TTI_SFPMAD(7, 5, 14, 7, 0);          /* B: p1 (c2 last use)  */  \
    TTI_SFPMAD(3, 1, 10, 1, 0);          /* A: p = p1*r + 1      */  \
    TTI_SFPMAD(7, 5, 10, 5, 0);          /* B: p                 */  \
    TTI_SFPSETEXP(0, 1, 2, 0);           /* A: E = p * 2^(k-127) */  \
    TTI_SFPSETEXP(0, 5, 6, 0);           /* B: E                 */  \
    TTI_SFPLOADI(3, 1, 0x3089);          /* q2 (shared)          */  \
    TTI_SFPLOADI(7, 1, 0xB71A);          /* q1 (shared)          */  \
    TTI_SFPMAD(3, 2, 7, 1, 0);           /* A: q2*E + q1         */  \
    TTI_SFPMAD(3, 6, 7, 5, 0);           /* B: q2*E + q1         */  \
    TTI_SFPMAD(1, 2, 10, 1, 0);          /* A: q = *E + 1        */  \
    TTI_SFPMAD(5, 6, 10, 5, 0);          /* B: q                 */  \
    TTI_SFPLOADMACRO(7, 0, 7, offA);     /* A: ld,relu,mad,store */  \
    TTI_SFPNOP;                          /* A swap busy          */  \
    TTI_SFPLOADI(7, 1, 0x3547);          /* next c2 (A mad cyc)  */  \
    TTI_SFPLOADMACRO(0, 0, 7, offB);     /* B: (seq0, lreg L0)   */  \
    TTI_SFPNOP;                          /* B swap / A store     */  \
    TTI_SFPLOAD(1, 0, 7, offNA);         /* next A x (B mad cyc) */  \
    TTI_SFPLOAD(5, 0, 7, offNB);         /* next B x             */  \
    TTI_SFPSETSGN(1, 1, 1, 1);           /* next A: n (B store)  */  \
    TTI_SFPSETSGN(1, 5, 5, 1);           /* next B: n = -|x|     */

inline void _calculate_softplus_bf16_fast_() {
    TTI_SFPLOAD(1, 0, 7, 0);
    TTI_SFPLOAD(5, 0, 7, 2);
    TTI_SFPLOADI(7, 1, 0x3547);  // c2 for the first pair
    TTI_SFPSETSGN(1, 1, 1, 1);
    TTI_SFPSETSGN(1, 5, 5, 1);
    SOFTPLUS_FAST_PAIR(0, 2, 4, 6)
    SOFTPLUS_FAST_PAIR(4, 6, 8, 10)
    SOFTPLUS_FAST_PAIR(8, 10, 12, 14)
    // last pair: no prefetch, but keep the final macro's store in-flight
    // window covered before returning to the wrapper's counter updates
    TTI_SFPMAD(1, 12, 13, 1, 0);
    TTI_SFPMAD(5, 12, 13, 5, 0);
    TTI_SFPSWAP(0, 1, 9, 1);
    TTI_SFPSWAP(0, 5, 9, 1);
    TTI_SFP_STOCH_RND(2, 0, 0, 1, 2, 2);
    TTI_SFP_STOCH_RND(2, 0, 0, 5, 6, 2);
    TTI_SFPCAST(2, 3, 0);
    TTI_SFPCAST(6, 0, 0);
    TTI_SFPMAD(3, 11, 1, 1, 0);
    TTI_SFPMAD(0, 11, 5, 5, 0);
    TTI_SFPMAD(7, 1, 14, 3, 0);
    TTI_SFPMAD(7, 5, 14, 7, 0);
    TTI_SFPMAD(3, 1, 10, 1, 0);
    TTI_SFPMAD(7, 5, 10, 5, 0);
    TTI_SFPSETEXP(0, 1, 2, 0);
    TTI_SFPSETEXP(0, 5, 6, 0);
    TTI_SFPLOADI(3, 1, 0x3089);
    TTI_SFPLOADI(7, 1, 0xB71A);
    TTI_SFPMAD(3, 2, 7, 1, 0);
    TTI_SFPMAD(3, 6, 7, 5, 0);
    TTI_SFPMAD(1, 2, 10, 1, 0);
    TTI_SFPMAD(5, 6, 10, 5, 0);
    TTI_SFPLOADMACRO(7, 0, 7, 12);
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPLOADMACRO(0, 0, 7, 14);
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

#undef SOFTPLUS_FAST_PAIR
#endif  // !DISABLE_SFPLOADMACRO

// Init reached through the bare dispatcher (SFPU_UNARY_INIT(softplus)) and by direct callers such as the tt-llk
// SDPA test: counters only, never the fast-path state.
inline void softplus_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

// Init for softplus_tile_init / softplus_tile_init_pack (SFPU_UNARY_INIT_FN): counters, plus -- for the bf16
// non-approx configuration calculate_softplus<false, false, 8> can serve with the fast kernel -- that kernel's
// state. The generic fallback (other template combinations, or beta/threshold other than 1.0/20.0 at runtime)
// reads none of that state.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void softplus_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
#ifndef DISABLE_SFPLOADMACRO
    if constexpr (!APPROXIMATION_MODE && !is_fp32_dest_acc_en) {
        _init_softplus_bf16_fast_();
    }
#endif
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void calculate_softplus_body(const float beta, const float beta_reciprocal, const float threshold) {
    sfpi::vFloat val = sfpi::dst_reg[0];
    sfpi::vFloat t = beta * val;

    v_if(t <= threshold) {
        // a = |t| via setsgn (clear sign bit, no branch)
        sfpi::vFloat a = sfpi::setsgn(t, 0);

#ifdef INP_FLOAT32
        // FP32: f(a) via degree-8 Horner on [0, 5]
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

        // Tail: f(a) ≈ exp(-a) for a > 5, via inline Cody-Waite exp +
        // 3-term Taylor ln(1+e) = e*(1 + e*(-1/2 + e/3))
        sfpi::vFloat neg_a = sfpi::setsgn(a, 1);
        v_if(a > SOFTPLUS_POLY_BOUNDARY) {
            sfpi::vFloat e = softplus_exp_negative(neg_a);
            residual = e * (1.0f + e * (-0.5f + e * 0.333333343f));
        }
        v_endif;
#else
        // BF16: f(a) via degree-6 Horner on [0, 5]
        sfpi::vFloat residual = PolynomialEvaluator::eval(
            a,
            SOFTPLUS_BF16_POLY_C0,
            SOFTPLUS_BF16_POLY_C1,
            SOFTPLUS_BF16_POLY_C2,
            SOFTPLUS_BF16_POLY_C3,
            SOFTPLUS_BF16_POLY_C4,
            SOFTPLUS_BF16_POLY_C5,
            SOFTPLUS_BF16_POLY_C6);

        // Tail: the degree-6 poly diverges past its [0, 5] fit domain, while the true
        // residual < exp(-5) = 0.0067 there. Clamping to 0 keeps softplus(t>0) = t within
        // bf16 rounding and avoids the ~8-op exp tail on every element.
        v_if(a > SOFTPLUS_POLY_BOUNDARY) { residual = 0.0f; }
        v_endif;
#endif

        // Reconstruct softplus(t):
        //   t >= 0: softplus(t) = t + f(t) = max(0,t) + residual
        //   t < 0:  softplus(t) = f(|t|) = 0 + residual
        t = sfpi::max(t, 0.0f);
        sfpi::vFloat sp = t + residual;

        // Round-to-nearest for bf16 destination (SFPSTORE defaults to truncation)
        sfpi::vFloat result = beta_reciprocal * sp;
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
    }
    v_endif;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_softplus(std::uint32_t param0, std::uint32_t param1, std::uint32_t param2) {
#ifndef DISABLE_SFPLOADMACRO
    if constexpr (!APPROXIMATION_MODE && !is_fp32_dest_acc_en && ITERATIONS == 8) {
        // The fast kernel is fitted and exhaustively validated for beta = 1.0 (and its reciprocal) with
        // threshold = 20.0 only; every other parameter set takes the generic code below.
        if (param0 == 0x3F800000u && param1 == 0x3F800000u && param2 == 0x41A00000u) {
            _calculate_softplus_bf16_fast_();
            return;
        }
    }
#endif
    const float beta = Converter::as_float(param0);
    const float beta_reciprocal = Converter::as_float(param1);
    const float threshold = Converter::as_float(param2);
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_softplus_body<APPROXIMATION_MODE, is_fp32_dest_acc_en>(beta, beta_reciprocal, threshold);
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
