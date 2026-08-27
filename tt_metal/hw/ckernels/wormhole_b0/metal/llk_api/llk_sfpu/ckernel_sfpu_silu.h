// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cmath_common.h"  // math::reset_counters, p_setrwc
#include "ckernel_sfpu_sigmoid.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel::sfpu {

// ======================================================================
// APPROXIMATION_MODE: 6-segment piecewise linear via SFPLUTFP32.
//
// silu(x) = x*sigmoid(x). Splitting off the linear part,
//
//   silu(x) = 0.5*x + E(|x|),   E(a) = a*(sigmoid(a) - 0.5)
//
// E is even, so it is exactly what the SGN_UPDATE form of the LUT produces, and
// this is the same decomposition (and the same loop shape) as gelu's approx
// path -- including the last segment carrying slope ~0.5, since E(a) -> 0.5a.
// The two halves then reconstruct correctly at both ends: for large positive x,
// 0.5x + 0.5x = x; for large negative x, 0.5x + 0.5|x| = 0.
//
// Getting to this form mattered for more than elegance. The direct spelling,
// x * (0.5 + copysgn(L(|x|), x)), needs x live *after* the LUT for the copysgn,
// and with six of the eight LRegs pinned to the table sfpi cannot allocate that
// ("cannot store sfpu register"). Folding |x|*L into the table itself leaves
// nothing but the LUT's own output needed after the instruction.
//
// Hardware breakpoints on |x| (FP16 6-entry TABLE2, sfpi mode = 0):
//   [0.0, 0.5): 0.092957*|x|
//   [0.5, 1.0): 0.339600*|x| - 0.114441
//   [1.0, 1.5): 0.490479*|x| - 0.262939
//   [1.5, 2.0): 0.570312*|x| - 0.380615
//   [2.0, 4.0): 0.583008*|x| - 0.397705
//   [4.0, inf): 0.507324*|x| - 0.078735
//
// Segment 0's intercept is pinned to 0 so silu(0) == 0 exactly. Max abs error
// 0.0225. Relative error is unbounded in the far-negative tail, where silu
// decays exponentially to 0 and a straight line cannot follow it -- callers
// needing that tail want the accurate path.
// ======================================================================
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_silu() {
    if constexpr (APPROXIMATION_MODE) {
        sfpi::vUInt l0 = sfpi::l_reg[sfpi::LRegs::LReg0];
        sfpi::vUInt l1 = sfpi::l_reg[sfpi::LRegs::LReg1];
        sfpi::vUInt l2 = sfpi::l_reg[sfpi::LRegs::LReg2];
        sfpi::vUInt l4 = sfpi::l_reg[sfpi::LRegs::LReg4];
        sfpi::vUInt l5 = sfpi::l_reg[sfpi::LRegs::LReg5];
        sfpi::vUInt l6 = sfpi::l_reg[sfpi::LRegs::LReg6];

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat in = sfpi::dst_reg[0];
            // 0.5 from a CREG: an inline 0.5f costs an SFPLOADI per iteration.
            sfpi::vFloat half = sfpi::vConstFloatPrgm0;
            sfpi::vFloat half_in = in * half;
            sfpi::vFloat result = lut2_sign(in, l0, l1, l2, l4, l5, l6, 0);
            sfpi::dst_reg[0] = half_in + result;
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

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // silu(x) = x * sigmoid(x)
        sfpi::vFloat result = x * _sfpu_sigmoid_<is_fp32_dest_acc_en>(x);

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
    if constexpr (APPROXIMATION_MODE) {
        // silu carries its own table (fit of E above), not sigmoid's.
        //
        // ORDER IS LOAD-BEARING: this CREG write must stay ahead of the table
        // stores. Programming a vConstFloatPrgm clobbers LReg0, which holds the
        // slopes for segments 0 and 1, so swapping these two blocks corrupts
        // exactly |x| < 1.0 and leaves |x| >= 1.0 intact -- a failure that reads
        // like a bad fit rather than a clobber. gelu_init has the same ordering.
        sfpi::vConstFloatPrgm0 = 0.5f;
        sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vUInt(0x356F2DF3);
        sfpi::l_reg[sfpi::LRegs::LReg1] = sfpi::vUInt(0x389037D9);
        sfpi::l_reg[sfpi::LRegs::LReg2] = sfpi::vUInt(0x380F38AA);
        sfpi::l_reg[sfpi::LRegs::LReg4] = sfpi::vUInt(0xAF537C00);
        sfpi::l_reg[sfpi::LRegs::LReg5] = sfpi::vUInt(0xB617B435);
        sfpi::l_reg[sfpi::LRegs::LReg6] = sfpi::vUInt(0xAD0AB65D);
    } else {
        // The accurate path goes through _sfpu_sigmoid_, which needs the
        // reciprocal constants seeded.
        sigmoid_init<false>();
    }
}

}  // namespace ckernel::sfpu
