// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"

// ckernel_sfpu_situ_glu.h builds on _sfpu_softcap_ and sfpi::approx_recip, neither of which
// has a Wormhole counterpart, so the API is Blackhole only.
#if defined(ARCH_BLACKHOLE)

#ifdef TRISC_MATH
#include "ckernel_sfpu_situ_glu.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
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
 * | vector_mode    | The vector mode of the operation                                       | int      | Must be one of the VectorMode values                  | False    |
 */
// clang-format on
ALWI void situ_glu_tile(uint32_t idst0, uint32_t idst1, uint32_t odst, VectorMode vector_mode = VectorMode::RC) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_situ_glu,
        (DST_ACCUM_MODE, 8 /* ITERATIONS */, sfpu::SituGluConfigKimi),
        idst0,
        idst1,
        odst,
        vector_mode)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void situ_glu_tile_init() { MATH((SFPU_BINARY_INIT_FN_NO_ARGS(situ_glu, sfpu::situ_glu_init))); }

}  // namespace ckernel

#endif  // ARCH_BLACKHOLE
