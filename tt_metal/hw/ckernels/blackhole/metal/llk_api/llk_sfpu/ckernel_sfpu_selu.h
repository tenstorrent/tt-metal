// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpu/ckernel_sfpu_expm1_cw.h"

namespace ckernel::sfpu {

// selu(x) = scale * x for x>=0, scale * alpha * (exp(x)-1) for x<0
// scale ≈ 1.0507, alpha ≈ 1.6733, scale*alpha ≈ 1.7581

// Programs the Cody-Waite constants of expm1_cw_clamped into vConstFloatPrgm0/1/2, which calculate_selu
// reads on every row. This kernel runs no reciprocal, so Prgm0 is not sfpu_reciprocal_init's 2.0f here.
inline void selu_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    expm1_cw_init_prgm_consts();
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_selu(uint32_t scale, uint32_t alpha) {
    const sfpi::vFloat scale_val = Converter::as_float(scale);
    const sfpi::vFloat scale_alpha = Converter::as_float(scale) * Converter::as_float(alpha);
    // expm1's top Horner coefficient, loaded once and held in an LReg for the whole loop (the three
    // Cody-Waite constants come from Prgm0/1/2, see the init above; sfpi 7.83.0 never hoists a literal out
    // of a loop by itself). With scale_val live as well there is no LReg for the second one, so it
    // stays a per-row literal.
    sfpi::vFloat h_top1 = CW_EXPM1_H_TOP1;
    constexpr float h_top0 = CW_EXPM1_H_TOP0;
// unroll 2: with expm1_cw_clamped inlined the loop body is large enough that
// partial unroll outperforms both full (unroll 8) and no-unroll (~0.8us on WH)
#pragma GCC unroll 2
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat result = scale_alpha * expm1_cw_clamped_prgm(x, h_top1, h_top0);

        v_if(x >= 0.0f) { result = scale_val * x; }
        v_endif;

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
