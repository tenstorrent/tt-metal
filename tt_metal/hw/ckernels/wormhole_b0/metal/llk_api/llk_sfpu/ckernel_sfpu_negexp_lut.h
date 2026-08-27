// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// ======================================================================
// exp(-|x|) as a 6-segment piecewise linear table for SFPLUTFP32.
//
// Shared by the approximate paths of ELU, SELU, CELU and xIELU: every one of
// them needs exp() only on the negative side, where it is bounded and decaying
// and so fits a straight-line table well. All four replace a full Cody-Waite
// range reduction plus a degree-4/5 polynomial (~16 SFPU ops) with one
// instruction.
//
// Hardware breakpoints on |x| (FP16 6-entry TABLE2, sfpi mode = 0):
//   [0.0, 0.5): -0.832031*|x| + 1.0
//   [0.5, 1.0): -0.477539*|x| + 0.837891
//   [1.0, 1.5): -0.289795*|x| + 0.653320
//   [1.5, 2.0): -0.175659*|x| + 0.483887
//   [2.0, 4.0): -0.058472*|x| + 0.238403
//   [4.0, inf):  0.006737
//
// Segment 0's intercept is pinned to exactly 1.0. That matters more than the
// accuracy it costs: the callers all reconstruct as alpha*(L - 1) + max(x, 0),
// so L(0) must be exactly 1 or the activation acquires a step at x = 0. A free
// fit puts 0.9878 there, which is a -0.0122*alpha discontinuity in ELU right at
// the join. Pinning moves max abs error on exp(-|x|) from 0.0139 to 0.0225 and
// removes the step.
//
// The table is only ever evaluated on the negative side by these callers, so its
// behaviour for positive arguments does not matter -- but note the table itself
// is even (SGN_UPDATE), which is exactly why feeding it min(x, 0) works: the
// hardware takes |min(x, 0)| = max(-x, 0).
// ======================================================================
inline void negexp_appx_load_lut() {
    sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vUInt(0xB7A4BAA8);
    sfpi::l_reg[sfpi::LRegs::LReg1] = sfpi::vUInt(0xB19FB4A3);
    sfpi::l_reg[sfpi::LRegs::LReg2] = sfpi::vUInt(0x7C00AB7C);
    sfpi::l_reg[sfpi::LRegs::LReg4] = sfpi::vUInt(0x3AB43C00);
    sfpi::l_reg[sfpi::LRegs::LReg5] = sfpi::vUInt(0x37BE393A);
    sfpi::l_reg[sfpi::LRegs::LReg6] = sfpi::vUInt(0x1EE633A1);
}

// Holds the six table registers across a loop. Declared as a struct so each
// caller's loop body stays the gelu-shaped "one value live across the LUT",
// which is what keeps these kernels inside sfpi's register budget.
struct NegExpLut {
    sfpi::vUInt l0, l1, l2, l4, l5, l6;

    sfpi_inline void load() {
        l0 = sfpi::l_reg[sfpi::LRegs::LReg0];
        l1 = sfpi::l_reg[sfpi::LRegs::LReg1];
        l2 = sfpi::l_reg[sfpi::LRegs::LReg2];
        l4 = sfpi::l_reg[sfpi::LRegs::LReg4];
        l5 = sfpi::l_reg[sfpi::LRegs::LReg5];
        l6 = sfpi::l_reg[sfpi::LRegs::LReg6];
    }

    sfpi_inline void store() {
        sfpi::l_reg[sfpi::LRegs::LReg0] = l0;
        sfpi::l_reg[sfpi::LRegs::LReg1] = l1;
        sfpi::l_reg[sfpi::LRegs::LReg2] = l2;
        sfpi::l_reg[sfpi::LRegs::LReg4] = l4;
        sfpi::l_reg[sfpi::LRegs::LReg5] = l5;
        sfpi::l_reg[sfpi::LRegs::LReg6] = l6;
    }

    // exp(-|v|). Callers pass min(x, 0) so this is exp(x) on the negative side
    // and exactly 1.0 for x >= 0.
    sfpi_inline sfpi::vFloat eval(sfpi::vFloat v) const { return lut2_sign(v, l0, l1, l2, l4, l5, l6, 0); }
};

}  // namespace ckernel::sfpu
