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
    sfpi::vFloat8Pair l0 = l_reg[sfpi::LRegs::LReg0];
    sfpi::vFloat8Pair l1 = l_reg[sfpi::LRegs::LReg1];
    sfpi::vFloat8Pair l2 = l_reg[sfpi::LRegs::LReg2];

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat val = dst_reg[0];

        dst_reg[0] = sfpi::lut(val, l0, l1, l2) + 0.5f;

        dst_reg++;
    }

    l_reg[LRegs::LReg0] = l0;
    l_reg[LRegs::LReg1] = l1;
    l_reg[LRegs::LReg2] = l2;
}

inline void sigmoid_appx_init() {
    // Load the 3 fp16b LUT coefficients into LReg0-2
    sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vFloat8Pair(0.22656f, 0.0f);
    sfpi::l_reg[sfpi::LRegs::LReg1] = sfpi::vFloat8Pair(0.26562f, -0.04687f);
    sfpi::l_reg[sfpi::LRegs::LReg2] = sfpi::vFloat8Pair(0.0f, 0.5f);
}

}  // namespace sfpu
}  // namespace ckernel
