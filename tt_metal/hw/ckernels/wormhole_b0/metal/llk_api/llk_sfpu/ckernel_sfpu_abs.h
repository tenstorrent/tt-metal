// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        dst_reg[0] = sfpi::abs(v);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs_int32() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        // int32 in dest is sign-magnitude on Wormhole, so abs() is just clearing the
        // sign bit (matches the original TTI_SFPABS path). sfpi::abs(vInt) would instead
        // do a two's-complement abs on the raw bits and produce 2^31 - |x| for x < 0.
        vInt v = dst_reg[0];
        dst_reg[0] = v & 0x7FFFFFFF;
        dst_reg++;
    }
}
}  // namespace sfpu
}  // namespace ckernel
