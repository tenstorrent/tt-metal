// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_polyval.h"

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

inline void softplus_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
#ifdef INP_FLOAT32
    sfpi::vConstFloatPrgm0 = SOFTPLUS_POLY_C6;
    sfpi::vConstFloatPrgm1 = SOFTPLUS_POLY_C7;
    sfpi::vConstFloatPrgm2 = SOFTPLUS_POLY_C8;
#else
    sfpi::vConstFloatPrgm0 = SOFTPLUS_BF16_POLY_C4;
    sfpi::vConstFloatPrgm1 = SOFTPLUS_BF16_POLY_C5;
    sfpi::vConstFloatPrgm2 = SOFTPLUS_BF16_POLY_C6;
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
            sfpi::vConstFloatPrgm0,
            sfpi::vConstFloatPrgm1,
            sfpi::vConstFloatPrgm2);

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
            sfpi::vConstFloatPrgm0,
            sfpi::vConstFloatPrgm1,
            sfpi::vConstFloatPrgm2);

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

struct SoftplusHandWords {
    std::uint32_t beta_lo, beta_hi, beta_one, thr_lo, thr_hi, thr_one, brec_lo, brec_hi, brec_one;
};
inline SoftplusHandWords softplus_hand_words(std::uint32_t b, std::uint32_t t, std::uint32_t r) {
    return {
        TT_OP_SFPLOADI(3, 2, b & 0xFFFF),
        TT_OP_SFPLOADI(3, 8, b >> 16),
        TT_OP_SFPLOADI(3, 0, b >> 16),
        TT_OP_SFPLOADI(3, 2, t & 0xFFFF),
        TT_OP_SFPLOADI(3, 8, t >> 16),
        TT_OP_SFPLOADI(3, 0, t >> 16),
        TT_OP_SFPLOADI(3, 2, r & 0xFFFF),
        TT_OP_SFPLOADI(3, 8, r >> 16),
        TT_OP_SFPLOADI(3, 0, r >> 16)};
}
template <int D>
inline void softplus_hand_row_bg(const SoftplusHandWords& w) {
    TTI_SFPLOAD(1, 0, ADDR_MOD_3, 2 * D);
    asm volatile("" ::: "memory");
    TT_INSN(w.beta_lo);
    asm volatile("" ::: "memory");
    asm volatile("" ::: "memory");
    TT_INSN(w.beta_hi);
    asm volatile("" ::: "memory");
    TTI_SFPMUL(1, 3, 9, 1, 0);
    asm volatile("" ::: "memory");
    TT_INSN(w.thr_lo);
    asm volatile("" ::: "memory");
    asm volatile("" ::: "memory");
    TT_INSN(w.thr_hi);
    asm volatile("" ::: "memory");
    if constexpr (D == 0) {
        TTI_REPLAY(0, 16, 1, 1);
        TTI_SFPMAD(11, 1, 3, 0, 0);
        TTI_SFPSETSGN(0, 1, 2, 1);
        TTI_SFPSETCC(0, 0, 0, 4);
        TTI_SFPMAD(2, 14, 13, 0, 0);
        TTI_SFPSWAP(0, 1, 9, 1);
        TTI_SFPMAD(2, 0, 12, 0, 0);
        TTI_SFPPUSHC(0, 0, 0, 0);
        TTI_SFPMAD(2, 0, 7, 0, 0);
        TTI_SFPLOADI(3, 0, 0x40a0);
        TTI_SFPMAD(2, 0, 6, 0, 0);
        TTI_SFPNOP;
        TTI_SFPMAD(2, 0, 5, 0, 0);
        TTI_SFPNOP;
        TTI_SFPMAD(2, 0, 4, 0, 0);
        TTI_SFPMAD(11, 2, 3, 2, 0);
        TTI_SFPADD(10, 1, 0, 0, 0);
    } else {
        TTI_REPLAY(0, 16, 0, 0);
    }
    TTI_SFPSETCC(0, 2, 0, 0);
    TTI_SFPMOV(0, 1, 0, 0);
    TTI_SFPPOPC(0, 0, 0, 0);
    asm volatile("" ::: "memory");
    TT_INSN(w.brec_lo);
    asm volatile("" ::: "memory");
    asm volatile("" ::: "memory");
    TT_INSN(w.brec_hi);
    asm volatile("" ::: "memory");
    TTI_SFPMUL(0, 3, 9, 0, 0);
    TTI_SFPNOP;
    TTI_SFP_STOCH_RND(0, 0, 9, 0, 0, 1);
    TTI_SFPSTORE(0, 0, ADDR_MOD_3, 2 * D);
    TTI_SFPENCC(3, 0, 0, 10);
}
template <int D>
inline void softplus_hand_row_bs(const SoftplusHandWords& w) {
    TTI_SFPLOAD(1, 0, ADDR_MOD_3, 2 * D);
    asm volatile("" ::: "memory");
    TT_INSN(w.beta_one);
    asm volatile("" ::: "memory");
    TTI_SFPMUL(1, 3, 9, 1, 0);
    asm volatile("" ::: "memory");
    TT_INSN(w.thr_one);
    asm volatile("" ::: "memory");
    TTI_SFPMAD(11, 1, 3, 0, 0);
    TTI_SFPSETSGN(0, 1, 2, 1);
    if constexpr (D == 0) {
        TTI_REPLAY(0, 16, 1, 1);
        TTI_SFPSETCC(0, 0, 0, 4);
        TTI_SFPMAD(2, 14, 13, 0, 0);
        TTI_SFPSWAP(0, 1, 9, 1);
        TTI_SFPMAD(2, 0, 12, 0, 0);
        TTI_SFPPUSHC(0, 0, 0, 0);
        TTI_SFPMAD(2, 0, 7, 0, 0);
        TTI_SFPLOADI(3, 0, 0x40a0);
        TTI_SFPMAD(2, 0, 6, 0, 0);
        TTI_SFPNOP;
        TTI_SFPMAD(2, 0, 5, 0, 0);
        TTI_SFPNOP;
        TTI_SFPMAD(2, 0, 4, 0, 0);
        TTI_SFPMAD(11, 2, 3, 2, 0);
        TTI_SFPADD(10, 1, 0, 0, 0);
        TTI_SFPSETCC(0, 2, 0, 0);
        TTI_SFPMOV(0, 1, 0, 0);
    } else {
        TTI_REPLAY(0, 16, 0, 0);
    }
    TTI_SFPPOPC(0, 0, 0, 0);
    asm volatile("" ::: "memory");
    TT_INSN(w.brec_one);
    asm volatile("" ::: "memory");
    TTI_SFPMUL(0, 3, 9, 0, 0);
    TTI_SFPNOP;
    TTI_SFP_STOCH_RND(0, 0, 9, 0, 0, 1);
    TTI_SFPSTORE(0, 0, ADDR_MOD_3, 2 * D);
    TTI_SFPENCC(3, 0, 0, 10);
}
template <int D>
inline void softplus_hand_row_bo(const SoftplusHandWords& w) {
    TTI_SFPLOAD(1, 0, ADDR_MOD_3, 2 * D);
    asm volatile("" ::: "memory");
    TT_INSN(w.thr_one);
    asm volatile("" ::: "memory");
    TTI_SFPMAD(11, 1, 3, 0, 0);
    TTI_SFPSETSGN(0, 1, 2, 1);
    TTI_SFPSETCC(0, 0, 0, 4);
    TTI_SFPMAD(2, 14, 13, 0, 0);
    if constexpr (D == 0) {
        TTI_REPLAY(0, 16, 1, 1);
        TTI_SFPSWAP(0, 1, 9, 1);
        TTI_SFPMAD(2, 0, 12, 0, 0);
        TTI_SFPPUSHC(0, 0, 0, 0);
        TTI_SFPMAD(2, 0, 7, 0, 0);
        TTI_SFPLOADI(3, 0, 0x40a0);
        TTI_SFPMAD(2, 0, 6, 0, 0);
        TTI_SFPNOP;
        TTI_SFPMAD(2, 0, 5, 0, 0);
        TTI_SFPNOP;
        TTI_SFPMAD(2, 0, 4, 0, 0);
        TTI_SFPMAD(11, 2, 3, 2, 0);
        TTI_SFPADD(10, 1, 0, 0, 0);
        TTI_SFPSETCC(0, 2, 0, 0);
        TTI_SFPMOV(0, 1, 0, 0);
        TTI_SFPPOPC(0, 0, 0, 0);
        TTI_SFP_STOCH_RND(0, 0, 9, 0, 0, 1);
    } else {
        TTI_REPLAY(0, 16, 0, 0);
    }
    TTI_SFPSTORE(0, 0, ADDR_MOD_3, 2 * D);
    TTI_SFPENCC(3, 0, 0, 10);
}
template <int D>
inline void softplus_hand_row_bn([[maybe_unused]] const SoftplusHandWords& w) {
    TTI_SFPLOAD(1, 0, ADDR_MOD_3, 2 * D);
    TTI_SFPSETSGN(0, 1, 2, 1);
    TTI_SFPMAD(2, 14, 13, 0, 0);
    if constexpr (D == 0) {
        TTI_REPLAY(0, 16, 1, 1);
        TTI_SFPSWAP(0, 1, 9, 1);
        TTI_SFPMAD(2, 0, 12, 0, 0);
        TTI_SFPLOADI(3, 0, 0x40a0);
        TTI_SFPMAD(2, 0, 7, 0, 0);
        TTI_SFPNOP;
        TTI_SFPMAD(2, 0, 6, 0, 0);
        TTI_SFPNOP;
        TTI_SFPMAD(2, 0, 5, 0, 0);
        TTI_SFPNOP;
        TTI_SFPMAD(2, 0, 4, 0, 0);
        TTI_SFPMAD(11, 2, 3, 2, 0);
        TTI_SFPADD(10, 1, 0, 0, 0);
        TTI_SFPSETCC(0, 2, 0, 0);
        TTI_SFPMOV(0, 1, 0, 0);
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFP_STOCH_RND(0, 0, 9, 0, 0, 1);
    } else {
        TTI_REPLAY(0, 16, 0, 0);
    }
    TTI_SFPSTORE(0, 0, ADDR_MOD_3, 2 * D);
}
template <int D>
inline void softplus_hand_row_fg(const SoftplusHandWords& w) {
    TTI_SFPLOAD(1, 0, ADDR_MOD_3, 2 * D);
    asm volatile("" ::: "memory");
    TT_INSN(w.beta_lo);
    asm volatile("" ::: "memory");
    asm volatile("" ::: "memory");
    TT_INSN(w.beta_hi);
    asm volatile("" ::: "memory");
    TTI_SFPMUL(1, 3, 9, 1, 0);
    asm volatile("" ::: "memory");
    TT_INSN(w.thr_lo);
    asm volatile("" ::: "memory");
    asm volatile("" ::: "memory");
    TT_INSN(w.thr_hi);
    asm volatile("" ::: "memory");
    if constexpr (D == 0) {
        TTI_REPLAY(0, 16, 1, 1);
        TTI_SFPMAD(11, 1, 3, 0, 0);
        TTI_SFPSETSGN(0, 1, 2, 1);
        TTI_SFPSETCC(0, 0, 0, 4);
        TTI_SFPMAD(2, 14, 13, 0, 0);
        TTI_SFPSWAP(0, 1, 9, 1);
        TTI_SFPMAD(2, 0, 12, 0, 0);
        TTI_SFPPUSHC(0, 0, 0, 0);
        TTI_SFPMAD(2, 0, 7, 0, 0);
        TTI_SFPLOADI(3, 0, 0x40a0);
        TTI_SFPMAD(2, 0, 6, 0, 0);
        TTI_SFPNOP;
        TTI_SFPMAD(2, 0, 5, 0, 0);
        TTI_SFPNOP;
        TTI_SFPMAD(2, 0, 4, 0, 0);
        TTI_SFPMAD(11, 2, 3, 2, 0);
        TTI_SFPADD(10, 1, 0, 0, 0);
    } else {
        TTI_REPLAY(0, 16, 0, 0);
    }
    TTI_SFPSETCC(0, 2, 0, 0);
    TTI_SFPMOV(0, 1, 0, 0);
    TTI_SFPPOPC(0, 0, 0, 0);
    asm volatile("" ::: "memory");
    TT_INSN(w.brec_lo);
    asm volatile("" ::: "memory");
    asm volatile("" ::: "memory");
    TT_INSN(w.brec_hi);
    asm volatile("" ::: "memory");
    TTI_SFPMUL(0, 3, 9, 0, 0);
    TTI_SFPNOP;
    TTI_SFPSTORE(0, 0, ADDR_MOD_3, 2 * D);
    TTI_SFPENCC(3, 0, 0, 10);
}
template <int D>
inline void softplus_hand_row_fs(const SoftplusHandWords& w) {
    TTI_SFPLOAD(1, 0, ADDR_MOD_3, 2 * D);
    asm volatile("" ::: "memory");
    TT_INSN(w.beta_one);
    asm volatile("" ::: "memory");
    TTI_SFPMUL(1, 3, 9, 1, 0);
    asm volatile("" ::: "memory");
    TT_INSN(w.thr_one);
    asm volatile("" ::: "memory");
    TTI_SFPMAD(11, 1, 3, 0, 0);
    TTI_SFPSETSGN(0, 1, 2, 1);
    if constexpr (D == 0) {
        TTI_REPLAY(0, 16, 1, 1);
        TTI_SFPSETCC(0, 0, 0, 4);
        TTI_SFPMAD(2, 14, 13, 0, 0);
        TTI_SFPSWAP(0, 1, 9, 1);
        TTI_SFPMAD(2, 0, 12, 0, 0);
        TTI_SFPPUSHC(0, 0, 0, 0);
        TTI_SFPMAD(2, 0, 7, 0, 0);
        TTI_SFPLOADI(3, 0, 0x40a0);
        TTI_SFPMAD(2, 0, 6, 0, 0);
        TTI_SFPNOP;
        TTI_SFPMAD(2, 0, 5, 0, 0);
        TTI_SFPNOP;
        TTI_SFPMAD(2, 0, 4, 0, 0);
        TTI_SFPMAD(11, 2, 3, 2, 0);
        TTI_SFPADD(10, 1, 0, 0, 0);
        TTI_SFPSETCC(0, 2, 0, 0);
        TTI_SFPMOV(0, 1, 0, 0);
    } else {
        TTI_REPLAY(0, 16, 0, 0);
    }
    TTI_SFPPOPC(0, 0, 0, 0);
    asm volatile("" ::: "memory");
    TT_INSN(w.brec_one);
    asm volatile("" ::: "memory");
    TTI_SFPMUL(0, 3, 9, 0, 0);
    TTI_SFPNOP;
    TTI_SFPSTORE(0, 0, ADDR_MOD_3, 2 * D);
    TTI_SFPENCC(3, 0, 0, 10);
}
template <int D>
inline void softplus_hand_row_fo(const SoftplusHandWords& w) {
    TTI_SFPLOAD(1, 0, ADDR_MOD_3, 2 * D);
    asm volatile("" ::: "memory");
    TT_INSN(w.thr_one);
    asm volatile("" ::: "memory");
    TTI_SFPMAD(11, 1, 3, 0, 0);
    TTI_SFPSETSGN(0, 1, 2, 1);
    TTI_SFPSETCC(0, 0, 0, 4);
    if constexpr (D == 0) {
        TTI_REPLAY(0, 16, 1, 1);
        TTI_SFPMAD(2, 14, 13, 0, 0);
        TTI_SFPSWAP(0, 1, 9, 1);
        TTI_SFPMAD(2, 0, 12, 0, 0);
        TTI_SFPPUSHC(0, 0, 0, 0);
        TTI_SFPMAD(2, 0, 7, 0, 0);
        TTI_SFPLOADI(3, 0, 0x40a0);
        TTI_SFPMAD(2, 0, 6, 0, 0);
        TTI_SFPNOP;
        TTI_SFPMAD(2, 0, 5, 0, 0);
        TTI_SFPNOP;
        TTI_SFPMAD(2, 0, 4, 0, 0);
        TTI_SFPMAD(11, 2, 3, 2, 0);
        TTI_SFPADD(10, 1, 0, 0, 0);
        TTI_SFPSETCC(0, 2, 0, 0);
        TTI_SFPMOV(0, 1, 0, 0);
        TTI_SFPPOPC(0, 0, 0, 0);
    } else {
        TTI_REPLAY(0, 16, 0, 0);
    }
    TTI_SFPSTORE(0, 0, ADDR_MOD_3, 2 * D);
    TTI_SFPENCC(3, 0, 0, 10);
}
template <int D>
inline void softplus_hand_row_fn([[maybe_unused]] const SoftplusHandWords& w) {
    TTI_SFPLOAD(1, 0, ADDR_MOD_3, 2 * D);
    TTI_SFPSETSGN(0, 1, 2, 1);
    if constexpr (D == 0) {
        TTI_REPLAY(0, 16, 1, 1);
        TTI_SFPMAD(2, 14, 13, 0, 0);
        TTI_SFPSWAP(0, 1, 9, 1);
        TTI_SFPMAD(2, 0, 12, 0, 0);
        TTI_SFPLOADI(3, 0, 0x40a0);
        TTI_SFPMAD(2, 0, 7, 0, 0);
        TTI_SFPNOP;
        TTI_SFPMAD(2, 0, 6, 0, 0);
        TTI_SFPNOP;
        TTI_SFPMAD(2, 0, 5, 0, 0);
        TTI_SFPNOP;
        TTI_SFPMAD(2, 0, 4, 0, 0);
        TTI_SFPMAD(11, 2, 3, 2, 0);
        TTI_SFPADD(10, 1, 0, 0, 0);
        TTI_SFPSETCC(0, 2, 0, 0);
        TTI_SFPMOV(0, 1, 0, 0);
        TTI_SFPENCC(3, 0, 0, 10);
    } else {
        TTI_REPLAY(0, 16, 0, 0);
    }
    TTI_SFPSTORE(0, 0, ADDR_MOD_3, 2 * D);
}
template <int D, int N, bool RND, int M>
inline void softplus_hand_rows(const SoftplusHandWords& w) {
    if constexpr (D < N) {
        if constexpr (RND == true && M == 0) {
            softplus_hand_row_bg<D>(w);
        } else if constexpr (RND == true && M == 1) {
            softplus_hand_row_bs<D>(w);
        } else if constexpr (RND == true && M == 2) {
            softplus_hand_row_bo<D>(w);
        } else if constexpr (RND == true && M == 3) {
            softplus_hand_row_bn<D>(w);
        } else if constexpr (RND == false && M == 0) {
            softplus_hand_row_fg<D>(w);
        } else if constexpr (RND == false && M == 1) {
            softplus_hand_row_fs<D>(w);
        } else if constexpr (RND == false && M == 2) {
            softplus_hand_row_fo<D>(w);
        } else if constexpr (RND == false && M == 3) {
            softplus_hand_row_fn<D>(w);
        }
        softplus_hand_rows<D + 1, N, RND, M>(w);
    }
}
template <int N>
inline void softplus_hand_advance() {
    if constexpr (N > 0) {
        TTI_INCRWC(0, 8, 0, 0);
        softplus_hand_advance<N - 1>();
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_softplus(std::uint32_t param0, std::uint32_t param1, std::uint32_t param2) {
#ifndef INP_FLOAT32
    if constexpr (ITERATIONS % 4 == 0) {
        const SoftplusHandWords w = softplus_hand_words(param0, param2, param1);
        if (param0 == 0x3f800000u && param1 == 0x3f800000u && param2 >= 0x40a00000u && param2 <= 0x7f800000u) {
            TTI_SFPLOADI(4, 2, 0xb9b4);
            TTI_SFPLOADI(4, 8, 0x3f31);
            TTI_SFPLOADI(5, 2, 0x6312);
            TTI_SFPLOADI(5, 8, 0xbf02);
            TTI_SFPLOADI(6, 2, 0x37cb);
            TTI_SFPLOADI(6, 8, 0x3e12);
            TTI_SFPLOADI(7, 2, 0x0067);
            TTI_SFPLOADI(7, 8, 0xbc55);
            softplus_hand_rows<0, ITERATIONS, !is_fp32_dest_acc_en, 3>(w);
        } else if (param0 == 0x3f800000u && param1 == 0x3f800000u && (param2 & 0xFFFF) == 0) {
            TTI_SFPLOADI(4, 2, 0xb9b4);
            TTI_SFPLOADI(4, 8, 0x3f31);
            TTI_SFPLOADI(5, 2, 0x6312);
            TTI_SFPLOADI(5, 8, 0xbf02);
            TTI_SFPLOADI(6, 2, 0x37cb);
            TTI_SFPLOADI(6, 8, 0x3e12);
            TTI_SFPLOADI(7, 2, 0x0067);
            TTI_SFPLOADI(7, 8, 0xbc55);
            softplus_hand_rows<0, ITERATIONS, !is_fp32_dest_acc_en, 2>(w);
        } else if (((param0 | param1 | param2) & 0xFFFF) == 0) {
            TTI_SFPLOADI(4, 2, 0xb9b4);
            TTI_SFPLOADI(4, 8, 0x3f31);
            TTI_SFPLOADI(5, 2, 0x6312);
            TTI_SFPLOADI(5, 8, 0xbf02);
            TTI_SFPLOADI(6, 2, 0x37cb);
            TTI_SFPLOADI(6, 8, 0x3e12);
            TTI_SFPLOADI(7, 2, 0x0067);
            TTI_SFPLOADI(7, 8, 0xbc55);
            softplus_hand_rows<0, ITERATIONS, !is_fp32_dest_acc_en, 1>(w);
        } else {
            TTI_SFPLOADI(4, 2, 0xb9b4);
            TTI_SFPLOADI(4, 8, 0x3f31);
            TTI_SFPLOADI(5, 2, 0x6312);
            TTI_SFPLOADI(5, 8, 0xbf02);
            TTI_SFPLOADI(6, 2, 0x37cb);
            TTI_SFPLOADI(6, 8, 0x3e12);
            TTI_SFPLOADI(7, 2, 0x0067);
            TTI_SFPLOADI(7, 8, 0xbc55);
            softplus_hand_rows<0, ITERATIONS, !is_fp32_dest_acc_en, 0>(w);
        }
        softplus_hand_advance<ITERATIONS / 4>();
        return;
    }
#else
    if (param0 == 0x3f800000u && param1 == 0x3f800000u) {
        if constexpr (ITERATIONS % 2 == 0) {
            TT_SFPLOADI(p_sfpu::LREG4, 2, param2 & 0xFFFF);
            TT_SFPLOADI(p_sfpu::LREG4, 8, param2 >> 16);
            TTI_SFPLOADI(0, 2, 0xaaab);
            TTI_SFPLOADI(0, 8, 0x3eaa);
            TTI_SFPCONFIG(0, 12, 0);
            TTI_SFPLOADI(0, 2, 0xaa3b);
            TTI_SFPLOADI(0, 8, 0x3fb8);
            TTI_SFPCONFIG(0, 13, 0);
            TTI_SFPLOAD(1, 0, 3, 0);
            TTI_SFPLOAD(2, 0, 3, 2);
            TTI_SFPSETSGN(0, 1, 1, 1);
            TTI_SFPSETSGN(0, 2, 2, 1);
            TTI_SFPLOADI(5, 2, 36525);
            TTI_SFPLOADI(5, 8, 14258);
            TTI_SFPMAD(1, 14, 5, 0, 0);
            TTI_SFPMAD(2, 14, 5, 3, 0);
            TTI_SFPLOADI(5, 2, 9009);
            TTI_SFPLOADI(5, 8, 47540);
            TTI_SFPMAD(1, 0, 5, 0, 0);
            TTI_SFPMAD(2, 3, 5, 3, 0);
            TTI_SFPLOADI(5, 2, 55633);
            TTI_SFPLOADI(5, 8, 15154);
            TTI_SFPMAD(1, 0, 5, 0, 0);
            TTI_SFPMAD(2, 3, 5, 3, 0);
            TTI_SFPLOADI(5, 2, 32564);
            TTI_SFPLOADI(5, 8, 48172);
            TTI_SFPMAD(1, 0, 5, 0, 0);
            TTI_SFPMAD(2, 3, 5, 3, 0);
            TTI_SFPLOADI(5, 2, 63622);
            TTI_SFPLOADI(5, 8, 15289);
            TTI_SFPMAD(1, 0, 5, 0, 0);
            TTI_SFPMAD(2, 3, 5, 3, 0);
            TTI_SFPLOADI(5, 2, 37777);
            TTI_SFPLOADI(5, 8, 15865);
            TTI_SFPMAD(1, 0, 5, 0, 0);
            TTI_SFPMAD(2, 3, 5, 3, 0);
            TTI_SFPLOADI(5, 2, 40788);
            TTI_SFPLOADI(5, 8, 48895);
            TTI_SFPMAD(1, 0, 5, 0, 0);
            TTI_SFPMAD(2, 3, 5, 3, 0);
            TTI_SFPLOADI(5, 2, 28510);
            TTI_SFPLOADI(5, 8, 16177);
            TTI_SFPMAD(1, 0, 5, 0, 0);
            TTI_SFPMAD(2, 3, 5, 3, 0);
            TTI_SFPSETSGN(1, 1, 7, 1);
            TTI_SFPLOADI(5, 0, 19264);
            TTI_SFPMAD(7, 13, 5, 1, 0);
            TTI_SFPLOADI(6, 2, 28672);
            TTI_SFPIADD(0, 1, 5, 6);
            TTI_SFPADDI(52032, 1, 0);
            TTI_SFPLOADI(6, 8, 48945);
            TTI_SFPMAD(1, 6, 7, 7, 0);
            TTI_SFPLOADI(6, 2, 65012);
            TTI_SFPLOADI(6, 8, 47109);
            TTI_SFPMAD(1, 6, 7, 6, 0);
            TTI_REPLAY(0, 16, 1, 1);
            TTI_SFPLOADI(1, 2, 3329);
            TTI_SFPLOADI(1, 8, 14672);
            TTI_SFPLOADI(7, 2, 2913);
            TTI_SFPLOADI(7, 8, 15030);
            TTI_SFPMAD(6, 1, 7, 1, 0);
            TTI_SFPLOADI(7, 2, 34953);
            TTI_SFPLOADI(7, 8, 15368);
            TTI_SFPMAD(6, 1, 7, 1, 0);
            TTI_SFPSETEXP(122, 12, 7, 1);
            TTI_SFPMAD(6, 1, 7, 1, 0);
            TTI_SFPSETEXP(124, 12, 7, 1);
            TTI_SFPMAD(6, 1, 7, 1, 0);
            TTI_SFPLOADI(7, 0, 16128);
            TTI_SFPMAD(6, 1, 7, 1, 0);
            TTI_SFPLOADI(7, 0, 48896);
            TTI_SFPMAD(6, 1, 10, 1, 0);
            TTI_SFPNOP;
            TTI_SFPMAD(6, 1, 10, 1, 0);
            TTI_SFPNOP;
            TTI_SFPEXEXP(0, 1, 6, 1);
            TTI_SFPIADD(0, 5, 6, 4);
            TTI_SFPSWAP(0, 6, 9, 1);
            TTI_SFPSETEXP(0, 1, 6, 0);
            TTI_SFPMAD(6, 12, 7, 1, 0);
            TTI_SFPLOAD(5, 0, 3, 0);
            TTI_SFPMAD(6, 1, 10, 1, 0);
            TTI_SFPSETSGN(0, 5, 7, 1);
            TTI_SFPMUL(6, 1, 9, 6, 0);
            TTI_SFPLOADI(1, 0, 16544);
            TTI_SFPMAD(11, 7, 1, 7, 0);
            TTI_SFPMAD(11, 5, 4, 1, 0);
            TTI_SFPSWAP(0, 5, 9, 1);
            TTI_SFPSETCC(0, 7, 0, 0);
            TTI_SFPMOV(0, 6, 0, 0);
            TTI_SFPENCC(3, 0, 0, 10);
            TTI_SFPADD(10, 5, 0, 5, 0);
            TTI_SFPSETCC(0, 1, 0, 4);
            TTI_SFPSTORE(5, 0, 3, 0);
            TTI_SFPENCC(3, 0, 0, 10);
            TTI_SFPSETSGN(1, 2, 7, 1);
            TTI_SFPLOADI(5, 0, 19264);
            TTI_SFPMAD(7, 13, 5, 1, 0);
            TTI_SFPLOADI(6, 2, 28672);
            TTI_SFPIADD(0, 1, 5, 6);
            TTI_SFPADDI(52032, 1, 0);
            TTI_SFPLOADI(6, 8, 48945);
            TTI_SFPMAD(1, 6, 7, 7, 0);
            TTI_SFPLOADI(6, 2, 65012);
            TTI_SFPLOADI(6, 8, 47109);
            TTI_SFPMAD(1, 6, 7, 6, 0);
            TTI_REPLAY(0, 16, 0, 0);
            TTI_SFPNOP;
            TTI_SFPMAD(6, 1, 10, 1, 0);
            TTI_SFPNOP;
            TTI_SFPEXEXP(0, 1, 6, 1);
            TTI_SFPIADD(0, 5, 6, 4);
            TTI_SFPSWAP(0, 6, 9, 1);
            TTI_SFPSETEXP(0, 1, 6, 0);
            TTI_SFPMAD(6, 12, 7, 1, 0);
            TTI_SFPLOAD(5, 0, 3, 2);
            TTI_SFPMAD(6, 1, 10, 1, 0);
            TTI_SFPSETSGN(0, 5, 7, 1);
            TTI_SFPMUL(6, 1, 9, 6, 0);
            TTI_SFPLOADI(1, 0, 16544);
            TTI_SFPMAD(11, 7, 1, 7, 0);
            TTI_SFPMAD(11, 5, 4, 1, 0);
            TTI_SFPSWAP(0, 5, 9, 1);
            TTI_SFPSETCC(0, 7, 0, 0);
            TTI_SFPMOV(0, 6, 3, 0);
            TTI_SFPENCC(3, 0, 0, 10);
            TTI_SFPADD(10, 5, 3, 5, 0);
            TTI_SFPSETCC(0, 1, 0, 4);
            TTI_SFPSTORE(5, 0, 3, 2);
            TTI_SFPENCC(3, 0, 0, 10);
            TTI_INCRWC(0, 4, 0, 0);
#pragma GCC unroll 0
            for (int d = 2; d < ITERATIONS; d += 2) {
                TTI_SFPLOAD(1, 0, 3, 0);
                TTI_SFPLOAD(2, 0, 3, 2);
                TTI_SFPSETSGN(0, 1, 1, 1);
                TTI_SFPSETSGN(0, 2, 2, 1);
                TTI_SFPLOADI(5, 2, 36525);
                TTI_SFPLOADI(5, 8, 14258);
                TTI_SFPMAD(1, 14, 5, 0, 0);
                TTI_SFPMAD(2, 14, 5, 3, 0);
                TTI_SFPLOADI(5, 2, 9009);
                TTI_SFPLOADI(5, 8, 47540);
                TTI_SFPMAD(1, 0, 5, 0, 0);
                TTI_SFPMAD(2, 3, 5, 3, 0);
                TTI_SFPLOADI(5, 2, 55633);
                TTI_SFPLOADI(5, 8, 15154);
                TTI_SFPMAD(1, 0, 5, 0, 0);
                TTI_SFPMAD(2, 3, 5, 3, 0);
                TTI_SFPLOADI(5, 2, 32564);
                TTI_SFPLOADI(5, 8, 48172);
                TTI_SFPMAD(1, 0, 5, 0, 0);
                TTI_SFPMAD(2, 3, 5, 3, 0);
                TTI_SFPLOADI(5, 2, 63622);
                TTI_SFPLOADI(5, 8, 15289);
                TTI_SFPMAD(1, 0, 5, 0, 0);
                TTI_SFPMAD(2, 3, 5, 3, 0);
                TTI_SFPLOADI(5, 2, 37777);
                TTI_SFPLOADI(5, 8, 15865);
                TTI_SFPMAD(1, 0, 5, 0, 0);
                TTI_SFPMAD(2, 3, 5, 3, 0);
                TTI_SFPLOADI(5, 2, 40788);
                TTI_SFPLOADI(5, 8, 48895);
                TTI_SFPMAD(1, 0, 5, 0, 0);
                TTI_SFPMAD(2, 3, 5, 3, 0);
                TTI_SFPLOADI(5, 2, 28510);
                TTI_SFPLOADI(5, 8, 16177);
                TTI_SFPMAD(1, 0, 5, 0, 0);
                TTI_SFPMAD(2, 3, 5, 3, 0);
                TTI_SFPSETSGN(1, 1, 7, 1);
                TTI_SFPLOADI(5, 0, 19264);
                TTI_SFPMAD(7, 13, 5, 1, 0);
                TTI_SFPLOADI(6, 2, 28672);
                TTI_SFPIADD(0, 1, 5, 6);
                TTI_SFPADDI(52032, 1, 0);
                TTI_SFPLOADI(6, 8, 48945);
                TTI_SFPMAD(1, 6, 7, 7, 0);
                TTI_SFPLOADI(6, 2, 65012);
                TTI_SFPLOADI(6, 8, 47109);
                TTI_SFPMAD(1, 6, 7, 6, 0);
                TTI_REPLAY(0, 16, 0, 0);
                TTI_SFPNOP;
                TTI_SFPMAD(6, 1, 10, 1, 0);
                TTI_SFPNOP;
                TTI_SFPEXEXP(0, 1, 6, 1);
                TTI_SFPIADD(0, 5, 6, 4);
                TTI_SFPSWAP(0, 6, 9, 1);
                TTI_SFPSETEXP(0, 1, 6, 0);
                TTI_SFPMAD(6, 12, 7, 1, 0);
                TTI_SFPLOAD(5, 0, 3, 0);
                TTI_SFPMAD(6, 1, 10, 1, 0);
                TTI_SFPSETSGN(0, 5, 7, 1);
                TTI_SFPMUL(6, 1, 9, 6, 0);
                TTI_SFPLOADI(1, 0, 16544);
                TTI_SFPMAD(11, 7, 1, 7, 0);
                TTI_SFPMAD(11, 5, 4, 1, 0);
                TTI_SFPSWAP(0, 5, 9, 1);
                TTI_SFPSETCC(0, 7, 0, 0);
                TTI_SFPMOV(0, 6, 0, 0);
                TTI_SFPENCC(3, 0, 0, 10);
                TTI_SFPADD(10, 5, 0, 5, 0);
                TTI_SFPSETCC(0, 1, 0, 4);
                TTI_SFPSTORE(5, 0, 3, 0);
                TTI_SFPENCC(3, 0, 0, 10);
                TTI_SFPSETSGN(1, 2, 7, 1);
                TTI_SFPLOADI(5, 0, 19264);
                TTI_SFPMAD(7, 13, 5, 1, 0);
                TTI_SFPLOADI(6, 2, 28672);
                TTI_SFPIADD(0, 1, 5, 6);
                TTI_SFPADDI(52032, 1, 0);
                TTI_SFPLOADI(6, 8, 48945);
                TTI_SFPMAD(1, 6, 7, 7, 0);
                TTI_SFPLOADI(6, 2, 65012);
                TTI_SFPLOADI(6, 8, 47109);
                TTI_SFPMAD(1, 6, 7, 6, 0);
                TTI_REPLAY(0, 16, 0, 0);
                TTI_SFPNOP;
                TTI_SFPMAD(6, 1, 10, 1, 0);
                TTI_SFPNOP;
                TTI_SFPEXEXP(0, 1, 6, 1);
                TTI_SFPIADD(0, 5, 6, 4);
                TTI_SFPSWAP(0, 6, 9, 1);
                TTI_SFPSETEXP(0, 1, 6, 0);
                TTI_SFPMAD(6, 12, 7, 1, 0);
                TTI_SFPLOAD(5, 0, 3, 2);
                TTI_SFPMAD(6, 1, 10, 1, 0);
                TTI_SFPSETSGN(0, 5, 7, 1);
                TTI_SFPMUL(6, 1, 9, 6, 0);
                TTI_SFPLOADI(1, 0, 16544);
                TTI_SFPMAD(11, 7, 1, 7, 0);
                TTI_SFPMAD(11, 5, 4, 1, 0);
                TTI_SFPSWAP(0, 5, 9, 1);
                TTI_SFPSETCC(0, 7, 0, 0);
                TTI_SFPMOV(0, 6, 3, 0);
                TTI_SFPENCC(3, 0, 0, 10);
                TTI_SFPADD(10, 5, 3, 5, 0);
                TTI_SFPSETCC(0, 1, 0, 4);
                TTI_SFPSTORE(5, 0, 3, 2);
                TTI_SFPENCC(3, 0, 0, 10);
                TTI_INCRWC(0, 4, 0, 0);
            }
            TTI_SFPLOADI(0, 2, 0x2331);
            TTI_SFPLOADI(0, 8, 0xb9b4);
            TTI_SFPCONFIG(0, 12, 0);
            TTI_SFPLOADI(0, 2, 0x8ead);
            TTI_SFPLOADI(0, 8, 0x37b2);
            TTI_SFPCONFIG(0, 13, 0);
            return;
        }
    }
#endif
    const float beta = Converter::as_float(param0);
    const float beta_reciprocal = Converter::as_float(param1);
    const float threshold = Converter::as_float(param2);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_softplus_body<APPROXIMATION_MODE, is_fp32_dest_acc_en>(beta, beta_reciprocal, threshold);
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
