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

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_sigmoid() {
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
