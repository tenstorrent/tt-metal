// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "ckernel_sfpu_sigmoid_appx.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel {
namespace sfpu {

// _sfpu_sigmoid_ with the exp's loop-invariant constants supplied by the caller, each an sfpi::vFloat built
// once before the row loop or a float literal: (c0, c1, c2) are read by the bf16 exp_21f arm, (neg_ln2_hi,
// p0, p1) by the fp32 Juffa arm; 1/ln2, which both arms read, comes from vConstFloatPrgm1, programmed by
// sigmoid_init<false>. Hoisting matters because sfpi 7.83.0 never lifts a literal out of a loop by itself,
// so each fp32 literal otherwise costs an SFPLOADI pair per row. Same arithmetic as the self-contained
// _sfpu_sigmoid_(x) below; only where the constants live differs.
template <bool is_fp32_acc_to_dest_mode, typename C0, typename C1, typename C2, typename H, typename P0, typename P1>
sfpi_inline sfpi::vFloat _sfpu_sigmoid_(sfpi::vFloat x, C0 c0, C1 c1, C2 c2, H neg_ln2_hi, P0 p0, P1 p1) {
    // Compute sigmoid as:
    // sigmoid(x) = 1 / (1 + exp(-x))

    sfpi::vFloat exp_neg_x;
    // If fp32 then use higher accuracy exp function
    // Otherwise, use exp_21f (~1 ULP on bfloat16)
    if constexpr (is_fp32_acc_to_dest_mode) {
        exp_neg_x = _sfpu_exp_fp32_accurate_prgm_<false>(-x, neg_ln2_hi, p0, p1);
    } else {
        exp_neg_x = _sfpu_exp_21f_bf16_prgm_<true>(-x, c0, c1, c2);
    }

    sfpi::vFloat denominator = 1.0f + exp_neg_x;

    sfpi::vFloat result;
    if constexpr (is_fp32_acc_to_dest_mode) {
        result = sfpu_reciprocal_iter<2>(denominator);
    } else {
        result = sfpu_reciprocal_iter<1>(denominator);
    }

    return result;
}

// Self-contained form: every constant is a literal, so it needs only sfpu_reciprocal_init's Prgm0 = 2.0f.
// Kept for the callers that evaluate one sigmoid outside a row loop (experimental clamped_silu,
// silu_scaled, the MoE gates); the row-loop kernels in this directory use the overload above.
template <bool is_fp32_acc_to_dest_mode = true>
sfpi_inline sfpi::vFloat _sfpu_sigmoid_(sfpi::vFloat x) {
    // Compute sigmoid as:
    // sigmoid(x) = 1 / (1 + exp(-x))

    sfpi::vFloat exp_neg_x;
    // If fp32 then use higher accuracy exp function
    // Otherwise, use exp_21f (~1 ULP on bfloat16)
    if constexpr (is_fp32_acc_to_dest_mode) {
        exp_neg_x = _sfpu_exp_accurate_<true>(-x);
    } else {
        exp_neg_x = _sfpu_exp_21f_bf16_<true>(-x);
    }

    sfpi::vFloat denominator = 1.0f + exp_neg_x;

    sfpi::vFloat result;
    if constexpr (is_fp32_acc_to_dest_mode) {
        result = sfpu_reciprocal_iter<2>(denominator);
    } else {
        result = sfpu_reciprocal_iter<1>(denominator);
    }

    return result;
}

// ---------------------------------------------------------------------------------------------------------------
// bf16-dest exact-path sigmoid (P7_SIGFAST). SIGMOID_BF16_IMPL selects the implementation calculate_sigmoid uses:
//   0 = the sfpi loop below (exp_21f + ARECIP + 1 guarded Newton step, constants hoisted)
//   3 = P5 "CB e2 + NR" (as SILU_BF16_IMPL 3 without the final multiply by x): xc = clamp(x, +-87.5);
//       f = -xc*log2e + 1.5*2^23; r = -xc*log2e - (f - M); 2^r ~ 1 + e1*r + e2*r^2; E = 2^r * 2^k by integer add
//       of (f << 23); y = ARECIP(1 + E); out = bf16(y + y*(1 - (1+E)*y)). <= 1 bf16 ulp from exact-rounded.
//       Software pipelined: the head of row i+1 runs in the latency slots of the tail of row i (19 slots per row).
//       Reads only Prgm1 (log2e, from sigmoid_init). The 87.5 clamp bound is loaded per row into a scratch LReg:
//       Prgm2 is written between sigmoid init and sigmoid by the MoE gate top-k kernels, and Prgm0 must stay 2.0.
// The fp32-dest path and the approximate path are unchanged.
// ---------------------------------------------------------------------------------------------------------------
#ifndef SIGMOID_BF16_IMPL
#define SIGMOID_BF16_IMPL 3
#endif

namespace sigmoid_detail {
constexpr std::uint32_t L0 = 0, L1 = 1, L2 = 2, L3 = 3, L4 = 4, L5 = 5, L6 = 6, L7 = 7;
constexpr std::uint32_t C0 = 9, C1 = 10, P1 = 13;  // 0.0, 1.0, Prgm1 = log2(e)
constexpr std::uint32_t SHFT_IMM_FROM_VC = 5;      // SFPSHFT ARG_IMM | ARG_IMM_USE_VC: VD = VC << imm
constexpr std::uint32_t IADD_CC_NONE = 4;          // SFPIADD: VD = VC + VD, lane flags untouched
constexpr std::uint32_t NEG_A = 1;                 // SFPMAD_MOD1_NEGATE_VA
constexpr std::uint16_t CLAMP_BF16 = 0x42af;       // 87.5f

// Constants: L1 = e2, L2 = e1, L4 = M = 1.5*2^23. Per-row registers: L3 = x -> xc, L7 = |x| -> r, L5 = clamp bound /
// f -> kk -> E -> d (carried to the next body), L0 = y -> km -> p, L6 = e -> out.
template <int ITERATIONS>
inline void calculate_sigmoid_bf16_p5_pipe() {
    TTI_SFPLOADI(L1, sfpi::SFPLOADI_MOD0_USHORT, 0x9e22);  // e2 = 0.239861041f (0x3E759E22)
    TTI_SFPLOADI(L1, sfpi::SFPLOADI_MOD0_UPPER, 0x3e75);
    TTI_SFPLOADI(L2, sfpi::SFPLOADI_MOD0_USHORT, 0xf3c2);  // e1 = 0.702938199f (0x3F33F3C2)
    TTI_SFPLOADI(L2, sfpi::SFPLOADI_MOD0_UPPER, 0x3f33);
    TTI_SFPLOADI(L4, sfpi::SFPLOADI_MOD0_FLOATB, 0x4b40);  // M = 12582912.0f
    // prologue: head of row 0
    TTI_SFPLOAD(L3, 0, ADDR_MOD_7, 0);
    TTI_SFPSETSGN(0, L3, L7, 1);  // |x|
    TTI_SFPLOADI(L5, sfpi::SFPLOADI_MOD0_FLOATB, CLAMP_BF16);
    TTI_SFPSWAP(0, L5, L7, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);  // min(|x|, 87.5)
    TTI_SFPSETSGN(0, L7, L3, 0);                             // xc
    TTI_SFPMAD(L3, P1, L4, L5, NEG_A);                       // f = -xc*log2e + M
    TTI_SFPMAD(L5, C1, L4, L0, NEG_A);                       // km = M - f
    TTI_SFPSHFT(23, L5, L5, SHFT_IMM_FROM_VC);               // kk = f << 23
    TTI_SFPMAD(L3, P1, L0, L7, NEG_A);                       // r = -xc*log2e + km
    TTI_SFPMAD(L7, L1, L2, L0, 0);                           // p = r*e2 + e1
    TTI_SFPMAD(L0, L7, C1, L0, 0);                           // p = p*r + 1
    TTI_SFPIADD(0, L0, L5, IADD_CC_NONE);                    // E = p + kk
    TTI_SFPADD(C1, L5, C1, L5, 0);                           // d = 1 + E
    if constexpr (ITERATIONS > 1) {
        constexpr int BODY = 19;
        TTI_REPLAY(0, BODY, 1, 1);
        TTI_SFPLOAD(L3, 0, ADDR_MOD_7, 2);                         // x' (row i+1)
        TTI_SFPARECIP(0, L5, L0, 0);                               // y = ~1/d (row i)
        TTI_SFPMAD(L5, L0, C1, L6, NEG_A);                         // e = 1 - d*y
        TTI_SFPSETSGN(0, L3, L7, 1);                               // |x'|
        TTI_SFPLOADI(L5, sfpi::SFPLOADI_MOD0_FLOATB, CLAMP_BF16);  // clamp bound (d is dead)
        TTI_SFPSWAP(0, L5, L7, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);    // min(|x'|, 87.5)
        TTI_SFPSETSGN(0, L7, L3, 0);                               // xc'
        TTI_SFPMAD(L3, P1, L4, L5, NEG_A);                         // f'
        TTI_SFPMAD(L0, L6, L0, L6, 0);                             // out = y*e + y
        TTI_SFPMAD(L5, C1, L4, L0, NEG_A);                         // km'
        TTI_SFPSHFT(23, L5, L5, SHFT_IMM_FROM_VC);                 // kk'
        TTI_SFPMAD(L3, P1, L0, L7, NEG_A);                         // r'
        TTI_SFP_STOCH_RND(0, 0, L0, L6, L6, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPMAD(L7, L1, L2, L0, 0);       // p'
        TTI_SFPSTORE(L6, 0, ADDR_MOD_7, 0);  // out (row i)
        TTI_SFPMAD(L0, L7, C1, L0, 0);       // p'
        TTI_INCRWC(0, 2, 0, 0);
        TTI_SFPIADD(0, L0, L5, IADD_CC_NONE);  // E'
        TTI_SFPADD(C1, L5, C1, L5, 0);         // d'
#pragma GCC unroll 8
        for (int i = 2; i < ITERATIONS; i++) {
            TTI_REPLAY(0, BODY, 0, 0);
        }
    }
    // epilogue: tail of the last row
    TTI_SFPARECIP(0, L5, L0, 0);
    TTI_SFPMAD(L5, L0, C1, L6, NEG_A);
    TTI_SFPMAD(L0, L6, L0, L6, 0);
    TTI_SFP_STOCH_RND(0, 0, L0, L6, L6, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPSTORE(L6, 0, ADDR_MOD_7, 0);
    TTI_INCRWC(0, 2, 0, 0);
}
}  // namespace sigmoid_detail

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_sigmoid() {
    if constexpr (!APPROXIMATION_MODE && !is_fp32_dest_acc_en && SIGMOID_BF16_IMPL == 3) {
        sigmoid_detail::calculate_sigmoid_bf16_p5_pipe<ITERATIONS>();
        return;
    }
    if constexpr (!APPROXIMATION_MODE) {
        // The exp's constants, loaded once and kept in LRegs for the whole loop; only the arm's own set is
        // hoisted (the other stays a float). Nothing but the row is live, so every constant fits.
        HoistedIf<!is_fp32_dest_acc_en> c0 = EXP_21F_C0, c1 = EXP_21F_C1, c2 = EXP_21F_C2;
        HoistedIf<is_fp32_dest_acc_en> neg_ln2_hi = EXP_FP32_NEG_LN2_HI, p0 = EXP_FP32_P0, p1 = EXP_FP32_P1;
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];
            sfpi::vFloat result = _sfpu_sigmoid_<is_fp32_dest_acc_en>(val, c0, c1, c2, neg_ln2_hi, p0, p1);
            if constexpr (!is_fp32_dest_acc_en) {
                result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
            }

            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
    } else {
        calculate_sigmoid_appx<ITERATIONS>();
    }
}

// The exact path's init contract, shared by silu_init and clamped_silu_glu_init (both forward here):
//   Prgm0 = 2.0f    (sfpu_reciprocal_init; the Newton step in sfpu_reciprocal_iter)
//   Prgm1 = 1/ln2   (both exps read it; 0x3FB8AA3B)
// Both must survive from this init to the calculate_* call. Fused kernels (deepseek_prefill fused_swiglu,
// the MoE gates, minimal_matmul) run one sigmoid/silu init before a loop that interleaves other SFPU
// code; they already keep Prgm0 intact for the reciprocal, and nothing they run writes LREG13 = Prgm1.
// Prgm2 is deliberately not part of the contract: the MoE gate topk code writes LREG14 between init and
// sigmoid (see _init_exp_hoisted_prgm_consts_). The approx path programs none of these.
template <bool APPROXIMATION_MODE>
inline void sigmoid_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    if constexpr (!APPROXIMATION_MODE) {
        sfpu_reciprocal_init<false>();
        _init_exp_hoisted_prgm_consts_();
    } else {
        sigmoid_appx_init();
    }
}

}  // namespace sfpu
}  // namespace ckernel
