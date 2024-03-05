// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

//TODO: Same as _calculate_sigmoid_, but different func name, find out where its used
template <bool APPROXIMATION_MODE, int ITERATIONS = 4>
inline void calculate_sigmoid_appx()
{
    // SFPU microcode
    vUInt l0 = l_reg[LRegs::LReg0];
    vUInt l1 = l_reg[LRegs::LReg1];
    vUInt l2 = l_reg[LRegs::LReg2];

    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];

        val = lut(val, l0, l1, l2);

        dst_reg[0] = val + 0.5f;

        dst_reg++;
    }

    l_reg[LRegs::LReg0] = l0;
    l_reg[LRegs::LReg1] = l1;
    l_reg[LRegs::LReg2] = l2;
}

template <bool APPROXIMATION_MODE>
inline void sigmoid_appx_init()
{
    uint imm0;
    uint imm1;
    uint imm2;
    imm0 = 0x3DFF;
    imm1 = 0x21D8;
    imm2 = 0xFF10;
    TTI_SFPLOADI(0, 2, imm0);
    TTI_SFPLOADI(1, 2, imm1);
    TTI_SFPLOADI(2, 2, imm2);
}

}  // namespace sfpu
}  // namespace ckernel
