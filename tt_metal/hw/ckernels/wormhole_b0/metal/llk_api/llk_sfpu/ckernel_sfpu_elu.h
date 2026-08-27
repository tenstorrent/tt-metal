// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpu/ckernel_sfpu_expm1_cw.h"
#include "ckernel_sfpu_negexp_lut.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE>
inline void elu_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_elu(std::uint32_t slope) {
    sfpi::vFloat alpha = Converter::as_float(slope);

    // APPROXIMATION_MODE: elu(x) = alpha*(exp(x) - 1) for x < 0, x for x >= 0.
    // Both branches collapse into one branch-free expression:
    //
    //   mx = max(x, 0)
    //   u  = mx - x            -- = |min(x, 0)|; the LUT only sees |u| anyway
    //   L  = exp(-|u|)         -- exactly 1.0 at u = 0 (segment 0 intercept pinned)
    //   t  = mx - alpha
    //   elu(x) = alpha*L + t
    //
    // x >= 0: u = 0, L = 1, t = x - alpha, so alpha + x - alpha = x.
    // x <  0: u = -x, L = exp(x), t = -alpha, so alpha*(exp(x) - 1).
    //
    // No v_if and only t live across the LUT, so this unrolls 8 where the accurate
    // path settles for 2 (expm1_cw_clamped inlines to ~16 ops).
    //
    // u is written mx - x, not sfpi::min(x, 0.0f): sfpi::min and sfpi::max each
    // lower to an SFPSWAP against the constant-zero LReg, and SFPSWAP writes both
    // its operands, so using both on one value has the first clobber the zero the
    // second reads. One SFPSWAP plus a subtract is also a shorter sequence.
    if constexpr (APPROXIMATION_MODE) {
        // ORDER IS LOAD-BEARING: programming a vConstFloatPrgm CREG clobbers
        // LReg0, so every CREG write must happen BEFORE the table is loaded.
        // LReg0 carries the slopes for segments 0 and 1, so getting this backwards
        // corrupts exactly |x| < 1.0 and leaves |x| >= 1.0 correct -- which reads
        // like a bad fit rather than a clobber. gelu_init and silu_init have the
        // same ordering for the same reason.
        //
        // The CREGs are programmed here rather than in <op>_init because the
        // coefficients are runtime kernel arguments; the six table stores are a
        // one-off outside the loop and cost nothing against 32 iterations.
        sfpi::vConstFloatPrgm0 = Converter::as_float(slope);
        negexp_appx_load_lut();

        sfpi::vUInt l0 = sfpi::l_reg[sfpi::LRegs::LReg0];
        sfpi::vUInt l1 = sfpi::l_reg[sfpi::LRegs::LReg1];
        sfpi::vUInt l2 = sfpi::l_reg[sfpi::LRegs::LReg2];
        sfpi::vUInt l4 = sfpi::l_reg[sfpi::LRegs::LReg4];
        sfpi::vUInt l5 = sfpi::l_reg[sfpi::LRegs::LReg5];
        sfpi::vUInt l6 = sfpi::l_reg[sfpi::LRegs::LReg6];

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat x = sfpi::dst_reg[0];
            sfpi::vFloat mx = sfpi::max(x, 0.0f);
            sfpi::vFloat u = mx - x;
            sfpi::vFloat t = mx - sfpi::vConstFloatPrgm0;
            sfpi::vFloat L = lut2_sign(u, l0, l1, l2, l4, l5, l6, 0);
            sfpi::dst_reg[0] = sfpi::vConstFloatPrgm0 * L + t;
            sfpi::dst_reg++;
        }

        sfpi::l_reg[sfpi::LRegs::LReg0] = l0;
        sfpi::l_reg[sfpi::LRegs::LReg1] = l1;
        sfpi::l_reg[sfpi::LRegs::LReg2] = l2;
        sfpi::l_reg[sfpi::LRegs::LReg4] = l4;
        sfpi::l_reg[sfpi::LRegs::LReg5] = l5;
        sfpi::l_reg[sfpi::LRegs::LReg6] = l6;
        return;
    }

// unroll 2: with expm1_cw_clamped inlined the loop body is large enough that
// partial unroll outperforms both full (unroll 8) and no-unroll (~0.8us on WH)
#pragma GCC unroll 2
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat result = alpha * expm1_cw_clamped(x);

        v_if(x >= 0.0f) { result = x; }
        v_endif;

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
