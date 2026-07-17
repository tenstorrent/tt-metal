// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_mish.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise mish activation:  mish(x) = x * tanh(softplus(x))
 * on each element of a tile in DST register at index tile_index.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Implemented as a single SFPU kernel via the algebraic identity
 *   mish(x) = x * u (u + 2) / (u^2 + 2u + 2),  where u = exp(x).
 *
 * In BH, to avoid catastrophic cancellation for sufficiently large positive x, we use
 *   For x >= 0, mish(x) = x * (1 - 2 / (u^2 + 2u + 2))
 *   For x <  0, mish(x) = x * u(u+2) / (u^2 + 2u + 2)
 *
 * For x >= 8.0, mish(x) is approximated as x.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool APPROXIMATION_MODE>
ALWI void mish_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_mish, (APPROXIMATION_MODE, DST_ACCUM_MODE), idst, VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool APPROXIMATION_MODE>
ALWI void mish_tile_init() {
    MATH(SFPU_UNARY_INIT_FN(mish, sfpu::mish_init, (APPROXIMATION_MODE, DST_ACCUM_MODE)));
}

}  // namespace ckernel
