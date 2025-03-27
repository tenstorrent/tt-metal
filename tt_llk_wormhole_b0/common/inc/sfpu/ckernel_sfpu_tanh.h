// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_load_config.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_tanh_(const int iterations)
{
    // SFPU microcode
    sfpi::vUInt l0 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vUInt l1 = sfpi::l_reg[sfpi::LRegs::LReg1];
    sfpi::vUInt l2 = sfpi::l_reg[sfpi::LRegs::LReg2];

#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat val = sfpi::dst_reg[0];
        val              = lut(val, l0, l1, l2);
        sfpi::dst_reg[0] = val;

        sfpi::dst_reg++;
    }

    sfpi::l_reg[sfpi::LRegs::LReg0] = l0;
    sfpi::l_reg[sfpi::LRegs::LReg1] = l1;
    sfpi::l_reg[sfpi::LRegs::LReg2] = l2;
}

template <bool APPROXIMATION_MODE>
inline void _init_tanh_()
{
    uint imm0;
    uint imm1;
    uint imm2;
    imm0 = 0x1DFF; // 0.90625*x
    imm1 = 0x481A; // 0.09375*x + 0.8125
    imm2 = 0xFF00; // 1
    _sfpu_load_imm16_(0, imm0);
    _sfpu_load_imm16_(1, imm1);
    _sfpu_load_imm16_(2, imm2);
}

} // namespace sfpu
} // namespace ckernel
