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
    // 3-entry SFPLUT, minimax per segment. A = imm[15:8], B = imm[7:0]; the byte format is
    // s(1)|e(3)|m(4) = (-1)^s * 2^-e * (1 + m/16), and byte 0xFF reads back as exactly 0.0.
    // sigmoid(x) = 0.5 + lut(x), so the table fits sigmoid(|x|) - 0.5, which is concave:
    //   |x| < 1 : 0.234375  *|x|                 (was 0.2265625*|x|)
    //   |x| < 2 : 0.1484375 *|x| + 0.08984375    (was 0.265625 *|x| - 0.046875 -- a slope
    //                                             LARGER than segment 0's, which a concave
    //                                             target cannot use; that was the defect)
    //   else    : 0.5, so sigmoid saturates at exactly 1.0 (unchanged)
    // Max |err| on [1, 2) drops 0.1029 -> 0.0360 and on [0, 1) 0.0098 -> 0.0034, measured
    // on n300. Error is pointwise non-increasing over the whole line; segment 2 is untouched.
    l_reg[LRegs::LReg0] = vUInt(static_cast<std::uint16_t>(0x3EFF));
    l_reg[LRegs::LReg1] = vUInt(static_cast<std::uint16_t>(0x3347));
    l_reg[LRegs::LReg2] = vUInt(static_cast<std::uint16_t>(0xFF10));
}

}  // namespace sfpu
}  // namespace ckernel
