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
inline void calculate_clamp(uint param0, uint param1, uint param2) {
    // All params are in FP16 format
    // param0 = min
    // param1 = max

    // uint format = (param0 >> 16)&0x1;
    s2vFloat16::Format format = s2vFloat16::fp16a;

    // SFPU microcode
    vFloat min = s2vFloat16(param0, format);
    vFloat max = s2vFloat16(param1, format);
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat val = dst_reg[0];

        v_if(val < min) { val = s2vFloat16(param0, format); }
        v_elseif(val >= max) { val = s2vFloat16(param1, format); }
        v_endif;

        dst_reg[0] = val + s2vFloat16b(param2);  // 12 bits

        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
