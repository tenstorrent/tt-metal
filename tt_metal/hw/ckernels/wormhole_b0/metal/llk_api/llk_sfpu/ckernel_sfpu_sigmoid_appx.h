// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// sigmoid(x) = 0.5 + g(x), where g is the odd function 0.5*tanh(x/2). g is
// fitted on |x| by a 6-segment SFPLUTFP32 table and re-signed with copysgn.
//
// This replaces the legacy 3-entry SFPLUT, which measures 0.121 max abs error on
// device; the 6-entry table measures ~0.018 by the same sweep. SFPLUTFP32 costs
// one cycle more than SFPLUT (2 vs 1) and the instruction count is unchanged.
template <int ITERATIONS = 8>
inline void calculate_sigmoid_appx() {
    vUInt l0 = l_reg[LRegs::LReg0];
    vUInt l1 = l_reg[LRegs::LReg1];
    vUInt l2 = l_reg[LRegs::LReg2];
    vUInt l4 = l_reg[LRegs::LReg4];
    vUInt l5 = l_reg[LRegs::LReg5];
    vUInt l6 = l_reg[LRegs::LReg6];

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat val = dst_reg[0];

        // mode 0 selects FP16 6-entry TABLE2, whose last breakpoint is 4.0
        // rather than TABLE1's 3.0 -- sigmoid is still moving at 3.0, so the
        // wider final segment fits it better (0.018 vs 0.047 max abs error).
        dst_reg[0] = copysgn(lut2_sign(val, l0, l1, l2, l4, l5, l6, 0), val) + 0.5f;

        dst_reg++;
    }

    l_reg[LRegs::LReg0] = l0;
    l_reg[LRegs::LReg1] = l1;
    l_reg[LRegs::LReg2] = l2;
    l_reg[LRegs::LReg4] = l4;
    l_reg[LRegs::LReg5] = l5;
    l_reg[LRegs::LReg6] = l6;
}

// Hardware breakpoints on |x| (FP16 6-entry TABLE2): 0.5, 1.0, 1.5, 2.0, 4.0.
//   [0.0, 0.5): 0.246216*|x|
//   [0.5, 1.0): 0.217163*|x| + 0.015091
//   [1.0, 1.5): 0.173096*|x| + 0.059418
//   [1.5, 2.0): 0.126465*|x| + 0.129272
//   [2.0, 4.0): 0.050629*|x| + 0.290283
//   [4.0, inf): 0.499756
// Segment 0's intercept is pinned to 0 so sigmoid(0) == 0.5 exactly. Slopes pack
// lo/hi into LReg0/1/2, intercepts into LReg4/5/6; 0x7C00 reads as 0.0.
inline void sigmoid_appx_init() {
    l_reg[LRegs::LReg0] = vUInt(0x32F333E1);
    l_reg[LRegs::LReg1] = vUInt(0x300C318A);
    l_reg[LRegs::LReg2] = vUInt(0x7C002A7B);
    l_reg[LRegs::LReg4] = vUInt(0x23BA7C00);
    l_reg[LRegs::LReg5] = vUInt(0x30232B9B);
    l_reg[LRegs::LReg6] = vUInt(0x37FF34A5);
}

}  // namespace sfpu
}  // namespace ckernel
