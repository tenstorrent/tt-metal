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

template <int ITERATIONS = 8>
inline void calculate_sigmoid_appx() {
    vUInt l0 = l_reg[LRegs::LReg0];
    vUInt l1 = l_reg[LRegs::LReg1];
    vUInt l2 = l_reg[LRegs::LReg2];

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat val = dst_reg[0];

        dst_reg[0] = lut(val, l0, l1, l2) + 0.5f;

        dst_reg++;
    }

    l_reg[LRegs::LReg0] = l0;
    l_reg[LRegs::LReg1] = l1;
    l_reg[LRegs::LReg2] = l2;
}

inline void sigmoid_appx_init() {
    // Load the 3 fp16b LUT coefficients into LReg0-2 via sfpi (each imm16 is loaded
    // as an unsigned 16-bit value, matching the original TTI_SFPLOADI mod0==2 path).
    l_reg[LRegs::LReg0] = vUInt(static_cast<std::uint16_t>(0x3DFF));
    l_reg[LRegs::LReg1] = vUInt(static_cast<std::uint16_t>(0x21D8));
    l_reg[LRegs::LReg2] = vUInt(static_cast<std::uint16_t>(0xFF10));
}

}  // namespace sfpu
}  // namespace ckernel
