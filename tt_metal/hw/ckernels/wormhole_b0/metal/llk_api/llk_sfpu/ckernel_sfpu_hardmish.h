// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel {
namespace sfpu {

// hardmish(x) = x * clamp(x + 2.8, 0.0, 5.0) / 5
//             = x * clamp(x + 2.8, 0.0, 5.0) * 0.2
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void hardmish() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat a = x + 2.8f;

        // sfpi::vec_min_max(a, b) puts min in a, max in b
        sfpi::vFloat low_bound = 0.0f;
        sfpi::vFloat high_bound = 5.0f;
        sfpi::vec_min_max(low_bound, a);   // a = max(a, 0.0)
        sfpi::vec_min_max(a, high_bound);  // a = min(a, 5.0)

        sfpi::dst_reg[0] = x * a * 0.2f;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
