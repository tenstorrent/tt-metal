// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "ckernel_sfpu_expm1_cw.h"

namespace ckernel::sfpu {

// selu(x) = scale * x for x>=0, scale * alpha * (exp(x)-1) for x<0
// scale ≈ 1.0507, alpha ≈ 1.6733, scale*alpha ≈ 1.7581

// Programs the Cody-Waite constants of expm1_cw_clamped into vConstFloatPrgm0/1/2, which calculate_selu
// reads on every row. This kernel runs no reciprocal, so Prgm0 is not sfpu_reciprocal_init's 2.0f here.
inline void selu_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    expm1_cw_init_prgm_consts();
}

bool bf16_dest_selu();
template <int ITERATIONS>
void calculate_selu_bf16();
// The scale the generated selu kernel is fitted for, as FP32 bits; any other scale runs the stock kernel.
inline constexpr std::uint32_t selu_bf16_scale_bits = 0x3f867d5fu;
// The alpha the generated selu kernel is fitted for, as FP32 bits; any other alpha runs the stock kernel.
inline constexpr std::uint32_t selu_bf16_alpha_bits = 0x3fd62d7du;
// Whether BF16 DEST runs the generated selu kernel as one call over the whole tile.
inline constexpr bool selu_bf16_whole_tile = true;

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_selu(uint32_t scale, uint32_t alpha) {
    if constexpr (!is_fp32_dest_acc_en && ITERATIONS == 32) {
        if (bf16_dest_selu() && scale == ckernel::sfpu::selu_bf16_scale_bits &&
            alpha == ckernel::sfpu::selu_bf16_alpha_bits) {
            calculate_selu_bf16<ITERATIONS>();
            return;
        }
    }
    const sfpi::vFloat scale_val = Converter::as_float(scale);
    const sfpi::vFloat scale_alpha = Converter::as_float(scale) * Converter::as_float(alpha);
    // expm1's top Horner coefficient, loaded once and held in an LReg for the whole loop (the three
    // Cody-Waite constants come from Prgm0/1/2, see the init above; sfpi (through 7.89.0) never hoists a literal out
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

#include "ckernel_sfpu_selu_bf16.h"
