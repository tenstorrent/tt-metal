// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpu/ckernel_sfpu_expm1_cw.h"
#include "ckernel_sfpu_negexp_lut.h"

namespace ckernel::sfpu {

// selu(x) = scale * x for x>=0, scale * alpha * (exp(x)-1) for x<0
// scale ≈ 1.0507, alpha ≈ 1.6733, scale*alpha ≈ 1.7581

// The approximate path programs its own CREGs + table inside calculate_selu().
template <bool APPROXIMATION_MODE>
inline void selu_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_selu(std::uint32_t scale, std::uint32_t alpha) {
    const sfpi::vFloat scale_val = Converter::as_float(scale);
    const sfpi::vFloat scale_alpha = Converter::as_float(scale) * Converter::as_float(alpha);

    // APPROXIMATION_MODE: same branch-free collapse as ELU (see
    // ckernel_sfpu_elu.h for the derivation and for why u is written mx - x),
    // with selu's two constants:
    //   t = scale*mx - scale_alpha;   selu(x) = scale_alpha*L + t
    if constexpr (APPROXIMATION_MODE) {
        // ORDER IS LOAD-BEARING: a vConstFloatPrgm write clobbers LReg0, which
        // holds the slopes for segments 0 and 1, so all CREG writes must precede
        // the table load. See ckernel_sfpu_elu.h for the full note.
        sfpi::vConstFloatPrgm0 = Converter::as_float(scale);
        sfpi::vConstFloatPrgm1 = Converter::as_float(scale) * Converter::as_float(alpha);
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
            sfpi::vFloat L = lut2_sign(u, l0, l1, l2, l4, l5, l6, 0);
            // One CREG per instruction. Written as a single
            //   t = Prgm0*mx - Prgm1;  result = Prgm1*L + t
            // sfpi has to materialise one of the two CREGs into an LReg to get
            // both into one SFPMAD, and with six LRegs pinned to the table that
            // spills. Splitting it as scale_alpha*(L - 1) + scale*mx keeps every
            // instruction down to a single CREG operand (1.0f is vConst1, free).
            sfpi::vFloat a = sfpi::vConstFloatPrgm1 * (L - 1.0f);
            sfpi::dst_reg[0] = sfpi::vConstFloatPrgm0 * mx + a;
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
        sfpi::vFloat result = scale_alpha * expm1_cw_clamped(x);

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
