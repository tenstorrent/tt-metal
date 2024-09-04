// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_tanh() {
    // SFPU microcode
    vUInt l0 = l_reg[LRegs::LReg0];
    vUInt l1 = l_reg[LRegs::LReg1];
    vUInt l2 = l_reg[LRegs::LReg2];

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat val = dst_reg[0];
        val = lut(val, l0, l1, l2);
        dst_reg[0] = val;

        dst_reg++;
    }

    l_reg[LRegs::LReg0] = l0;
    l_reg[LRegs::LReg1] = l1;
    l_reg[LRegs::LReg2] = l2;
}

template <bool APPROXIMATION_MODE>
inline void tanh_init() {
    uint imm0;
    uint imm1;
    uint imm2;
    imm0 = 0x1DFF;  // 0.90625*x
    imm1 = 0x481A;  // 0.09375*x + 0.8125
    imm2 = 0xFF00;  // 1
    _sfpu_load_imm16_(0, imm0);
    _sfpu_load_imm16_(1, imm1);
    _sfpu_load_imm16_(2, imm2);
}

}  // namespace sfpu
}  // namespace ckernel
