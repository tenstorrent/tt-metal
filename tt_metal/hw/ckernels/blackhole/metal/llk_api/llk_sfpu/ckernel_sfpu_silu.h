// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cmath_common.h"  // math::reset_counters, p_setrwc
#include "ckernel_sfpu_sigmoid.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel::sfpu {

template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_silu() {
    // The exp's constants, loaded once and kept in LRegs for the whole loop (see _sfpu_sigmoid_prgm_); 1/ln2 is
    // Prgm1 from silu_init. x stays live across the sigmoid, so the fp32 arm has room for two of its
    // three: p1 stays a per-row literal.
    HoistedIf<!is_fp32_dest_acc_en> c0 = EXP_21F_C0, c1 = EXP_21F_C1, c2 = EXP_21F_C2;
    HoistedIf<is_fp32_dest_acc_en> neg_ln2_hi = EXP_FP32_NEG_LN2_HI, p0 = EXP_FP32_P0;
    constexpr float p1 = EXP_FP32_P1;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // silu(x) = x * sigmoid(x). For x <= -88.5f (including -inf), exp(-x)
        // overflows so sigmoid(x) underflows to 0 and the product would be
        // (-inf) * 0.0f = NaN.
        //
        // Clamping the input is enough, because the NaN needs an *infinite*
        // x: with x finite, x * 0.0f is -0.0f, not NaN. For x = -88.5f,
        // sigmoid is ~3.7e-39 and the product is ~-3.3e-37, which rounds to
        // zero in every destination format — and if exp underflows all the
        // way to 0 the product is -0.0f, which is equally correct.
        //
        // Branchless on purpose: a v_if/v_else here measured ~130 cycles/tile
        // slower on Blackhole and it costs a live value in an arm that, per
        // the note above, only has LRegs for two of its three exp constants.
        // sfpi::max is the same guard expm1_cw_clamped uses.
        const sfpi::vFloat xc = sfpi::max(x, -88.5f);
        sfpi::vFloat result = xc * _sfpu_sigmoid_prgm_<is_fp32_dest_acc_en>(xc, c0, c1, c2, neg_ln2_hi, p0, p1);

        // Round to bfloat16 if not in fp32 accumulation mode
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void silu_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    // calculate_silu uses the non-approx sigmoid path via _sfpu_sigmoid_prgm_, so we must use non-approx
    // sigmoid_init: it programs Prgm0 = 2.0f for the reciprocal and Prgm1 = 1/ln2 for the exp.
    sigmoid_init<false>();
}

}  // namespace ckernel::sfpu
