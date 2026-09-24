// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"

// ckernel_sfpu_situ_glu.h builds on _sfpu_softcap_ and sfpi::approx_recip, neither of which
// has a Wormhole counterpart, so the API is Blackhole only.
#if defined(ARCH_BLACKHOLE)

#ifdef TRISC_MATH
#include "ckernel_sfpu_situ_glu.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs the element-wise SiTU-GLU activation over the tiles at idst0 (gate) and idst1 (up),
 * writing the result to odst in DST:
 *
 *   situ_a  = beta_gate * tanh(gate / beta_gate) * sigmoid(gate)
 *   up_half = beta_up   * tanh(up   / beta_up)
 *   odst    = situ_a * up_half
 *
 * Both halves are bounded, so |odst| <= beta_gate * beta_up, up to the rounding of the two
 * halves and of the packed result. The betas are the compile-time
 * Kimi K3 values (4 for the gate half, 25 for the up half); other models add a config next to
 * ckernel::sfpu::SituGluConfigKimi and call ckernel::sfpu::calculate_situ_glu directly.
 *
 * Both operands stay in DST, so no intermediate is materialized to L1 or DRAM. The DST register
 * buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                            | Type     | Valid Range                                           | Required |
 * |----------------|------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0          | The index of the tile in DST register buffer holding the gate operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1          | The index of the tile in DST register buffer holding the up operand    | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst           | The index of the tile in DST register buffer to use as output          | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void situ_glu_tile(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    MATH((sfpu::SituGlu<DST_ACCUM_MODE, 8 /* ITERATIONS */, sfpu::SituGluConfigKimi>::run(idst0, idst1, odst)));
}

/**
 * Legacy overload selecting the faces to process with a VectorMode. Prefer the overload above, which
 * processes the full tile.
 */
ALWI void situ_glu_tile(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst, VectorMode vector_mode) {
    MATH((sfpu::SituGlu<DST_ACCUM_MODE, 8 /* ITERATIONS */, sfpu::SituGluConfigKimi>::run_vector_mode(
        vector_mode, idst0, idst1, odst)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void situ_glu_tile_init() { MATH((sfpu::SituGlu<DST_ACCUM_MODE>::init())); }

}  // namespace ckernel

#endif  // ARCH_BLACKHOLE
