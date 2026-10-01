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

// sigmoid(x) = 1 / (1 + exp(-x)) with the exp's constants supplied by the caller. Each is a float (materialised
// by SFPLOADI where used, exactly as a literal is), an sfpi::vFloat built once before a row loop, or a
// vConstFloatPrgmN: one_ln2 is read by both exp arms, (c0, c1, c2) by the bf16 exp_21f arm and
// (neg_ln2_hi, p0, p1) by the fp32 Juffa arm. Hoisting matters because sfpi (through 7.84.0) never lifts a
// literal out of a loop by itself, so each fp32 literal otherwise costs an SFPLOADI pair per row.
// _sfpu_sigmoid_prgm_ and _sfpu_sigmoid_ below are thin wrappers over this one body.
template <
    bool is_fp32_acc_to_dest_mode,
    typename K,
    typename C0,
    typename C1,
    typename C2,
    typename H,
    typename P0,
    typename P1>
sfpi_inline sfpi::vFloat _sfpu_sigmoid_core_(
    sfpi::vFloat x, K one_ln2, C0 c0, C1 c1, C2 c2, H neg_ln2_hi, P0 p0, P1 p1) {
    sfpi::vFloat exp_neg_x;
    // If fp32 then use higher accuracy exp function
    // Otherwise, use exp_21f (~1 ULP on bfloat16)
    if constexpr (is_fp32_acc_to_dest_mode) {
        exp_neg_x = _sfpu_exp_fp32_accurate_<false /*unsafe*/>(-x, one_ln2, neg_ln2_hi, p0, p1);
    } else {
        exp_neg_x = _sfpu_exp_21f_bf16_<true /*is_fp32_dest_acc_en*/>(-x, one_ln2, c0, c1, c2);
    }

    sfpi::vFloat denominator = 1.0f + exp_neg_x;

    sfpi::vFloat result;
    if constexpr (is_fp32_acc_to_dest_mode) {
        result = sfpu_reciprocal_iter<2 /*max_iter*/>(denominator);
    } else {
        result = sfpu_reciprocal_iter<1 /*max_iter*/>(denominator);
    }

    return result;
}

// Row-loop form: 1/ln2 from vConstFloatPrgm1 (programmed by sigmoid_init<false>, which the caller's init
// must have run), the other exp constants from the caller (see _sfpu_sigmoid_core_).
template <bool is_fp32_acc_to_dest_mode, typename C0, typename C1, typename C2, typename H, typename P0, typename P1>
sfpi_inline sfpi::vFloat _sfpu_sigmoid_prgm_(sfpi::vFloat x, C0 c0, C1 c1, C2 c2, H neg_ln2_hi, P0 p0, P1 p1) {
    return _sfpu_sigmoid_core_<is_fp32_acc_to_dest_mode>(x, sfpi::vConstFloatPrgm1, c0, c1, c2, neg_ln2_hi, p0, p1);
}

// Self-contained form: every constant is a literal, so it needs only sfpu_reciprocal_init's Prgm0 = 2.0f.
// Not yet migrated to the row-loop form: calculate_clamped_silu_glu and the experimental clamped_silu /
// silu_scaled kernels (they need only Prgm0).
template <bool is_fp32_acc_to_dest_mode = true>
sfpi_inline sfpi::vFloat _sfpu_sigmoid_(sfpi::vFloat x) {
    return _sfpu_sigmoid_core_<is_fp32_acc_to_dest_mode>(
        x, EXP_21F_ONE_LN2, EXP_21F_C0, EXP_21F_C1, EXP_21F_C2, EXP_FP32_NEG_LN2_HI, EXP_FP32_P0, EXP_FP32_P1);
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
            sfpi::vFloat result = _sfpu_sigmoid_prgm_<is_fp32_dest_acc_en>(val, c0, c1, c2, neg_ln2_hi, p0, p1);
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

// The exact path's init contract for calculate_sigmoid, and for calculate_silu (silu_init forwards here):
//   Prgm0 = 2.0f    (sfpu_reciprocal_init; the Newton step in sfpu_reciprocal_iter)
//   Prgm1 = 1/ln2   (both exps read it; 0x3FB8AA3B)
// Both must survive from this init to the calculate_* call. (clamped_silu_glu_init also forwards here, but
// calculate_clamped_silu_glu uses the literal _sfpu_sigmoid_(x) and reads only Prgm0.)
// Fused kernels (deepseek_prefill fused_swiglu, the MoE gates, minimal_matmul) run one sigmoid/silu init
// before a loop that interleaves other SFPU code; they already keep Prgm0 intact for the reciprocal, and
// nothing they run writes LREG13 = Prgm1.
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
