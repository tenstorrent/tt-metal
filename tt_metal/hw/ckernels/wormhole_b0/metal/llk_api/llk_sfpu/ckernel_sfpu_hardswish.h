// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_converter.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

using namespace sfpi;

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void calculate_hardswish() {
    // hardswish(x) = x * max(0, min(1, x/6 + 0.5))
    //              = x * hardsigmoid(x)
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x = dst_reg[0];
        vFloat a = x * 0.16666667f + 0.5f;  // x/6 + 0.5
        vFloat low = 0.0f;
        vec_min_max(low, a);  // a = max(a, 0.0); low = min(a, 0.0)
        vFloat high = 1.0f;
        vec_min_max(a, high);  // a = min(a, 1.0); high = max(a, 1.0)
        dst_reg[0] = x * a;    // x * hardsigmoid(x)
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
