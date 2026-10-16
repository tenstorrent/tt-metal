// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

/**
 * @brief Element-wise x + 1 over a tile (toy kernel for the fuser knowledge-transfer demo).
 *
 * The plain sfpi store writes through the default (no-increment) address mode and dst_reg++
 * advances the Dest counter by one SFP row pair, so only the shared SFPU init is needed (no
 * op-specific ADDR_MOD).
 *
 * @tparam APPROXIMATION_MODE: Unused (the add is exact); retained for dispatcher signature symmetry.
 * @tparam ITERATIONS: Number of SFP-row pairs to process (8 for a 32x16 face).
 */
template <bool APPROXIMATION_MODE, int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_add_one() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = val + 1.0f;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
