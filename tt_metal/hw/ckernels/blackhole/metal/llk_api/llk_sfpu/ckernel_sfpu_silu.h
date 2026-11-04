// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cmath_common.h"  // math::reset_counters, p_setrwc
#include "ckernel_sfpu_sigmoid.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel::sfpu {

bool bf16_dest_silu();
template <int ITERATIONS>
void calculate_silu_bf16();
// Whether BF16 DEST runs the generated silu kernel as one call over the whole tile.
inline constexpr bool silu_bf16_whole_tile = true;

template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_silu() {
    if constexpr (!is_fp32_dest_acc_en && ITERATIONS == 32) {
        if (bf16_dest_silu()) {
            calculate_silu_bf16<ITERATIONS>();
            return;
        }
    }
    // The exp's constants, loaded once and kept in LRegs for the whole loop (see _sfpu_sigmoid_prgm_); 1/ln2 is
    // Prgm1 from silu_init. x stays live across the sigmoid, so the fp32 arm has room for two of its
    // three: p1 stays a per-row literal.
    HoistedIf<!is_fp32_dest_acc_en> c0 = EXP_21F_C0, c1 = EXP_21F_C1, c2 = EXP_21F_C2;
    HoistedIf<is_fp32_dest_acc_en> neg_ln2_hi = EXP_FP32_NEG_LN2_HI, p0 = EXP_FP32_P0;
    constexpr float p1 = EXP_FP32_P1;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // silu(x) = x * sigmoid(x)
        sfpi::vFloat result = x * _sfpu_sigmoid_prgm_<is_fp32_dest_acc_en>(x, c0, c1, c2, neg_ln2_hi, p0, p1);

        // Round to bfloat16 if not in fp32 accumulation mode
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

void init_silu_bf16();

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = true>
inline void silu_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    // calculate_silu uses the non-approx sigmoid path via _sfpu_sigmoid_prgm_, so we must use non-approx
    // sigmoid_init: it programs Prgm0 = 2.0f for the reciprocal and Prgm1 = 1/ln2 for the exp.
    sigmoid_init<false>();
    if constexpr (!is_fp32_dest_acc_en) {
        init_silu_bf16();
    }
}

}  // namespace ckernel::sfpu

#include "ckernel_sfpu_silu_bf16.h"
