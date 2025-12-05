// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void hardmish() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat a = x + 2.8f;

        // Branchless clamp using vec_min_max:
        // vec_min_max(a, b) puts min in a, max in b
        sfpi::vFloat low = 0.f;
        sfpi::vFloat high = 5.0f;
        sfpi::vec_min_max(low, a);   // a = max(a, 0.0)
        sfpi::vec_min_max(a, high);  // a = min(a, 5.0)

        sfpi::dst_reg[0] = x * a * 0.2f;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
