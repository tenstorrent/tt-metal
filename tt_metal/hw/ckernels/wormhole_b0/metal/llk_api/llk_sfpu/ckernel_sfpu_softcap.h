// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// softcap(x, cap) = cap * tanh(x / cap)
//
// This kernel handles the pre- and post-scaling around hardware tanh:
//   Step 1 (pre_tanh):  x <- x * (1/cap)
//   Step 2 (external):  hardware tanh_tile (called at compute API level)
//   Step 3 (post_tanh): x <- x * cap
//
// The cap parameter is passed as a uint32_t encoding of a float in FP16_B format
// and decoded with sfpi::s2vFloat16b().

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softcap_pre_tanh(const uint32_t param0) {
    // param0 = 1.0f / cap, encoded as FP16_B
    sfpi::vFloat inv_cap = sfpi::s2vFloat16b(param0);

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = x * inv_cap;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softcap_post_tanh(const uint32_t param0) {
    // param0 = cap, encoded as FP16_B
    sfpi::vFloat cap = sfpi::s2vFloat16b(param0);

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = x * cap;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
