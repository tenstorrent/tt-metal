// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_softplus.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_logsigmoid() {
    // logsigmoid(x) = -softplus(-x)
    #pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x = dst_reg[0];
        dst_reg[0] = -softplus(-x);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void logsigmoid_init() {
    softplus_init<APPROXIMATION_MODE>();
}

}  // namespace sfpu
}  // namespace ckernel
