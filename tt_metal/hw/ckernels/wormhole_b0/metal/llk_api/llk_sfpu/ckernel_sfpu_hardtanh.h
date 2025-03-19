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

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_hardtanh(uint param0, uint param1, uint param2) {
    // All params are in FP16_B format
    // param0 = -(neg_threshold)
    // param1 = -(pos_threshold - neg_threshold)
    // param2 = -(pos_threshold)

    vFloat p0 = s2vFloat16(param0);
    vFloat p1 = s2vFloat16(param1);
    vFloat p2 = s2vFloat16(param2);
// SFPU microcode
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat val = dst_reg[0];

        val += p0;  // 12 bits
        v_if(val < 0.0f) { val = 0.0f; }
        v_endif;

        val += p1;  // 12 bits
        v_if(val >= 0.0f) { val = 0.0f; }
        v_endif;

        val += p2;  // 12 bits

        dst_reg[0] = val;

        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
