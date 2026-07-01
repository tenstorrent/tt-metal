// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_rpow.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */
ALWI void rpow_tile_init() { MATH(SFPU_UNARY_INIT_FN(rpow, sfpu::sfpu_binary_pow_init, (APPROX))); }
// clang-format off
/**
 * Performs element-wise computation of the rpow on each element of a tile
 * where rpow(exponent, scalar_base) = pow(scalar_base, exponent)
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                 | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the computation on  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | base_val       | The base value to raise to the power of each element in the tile            | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void rpow_tile(uint32_t idst, uint32_t base_val, VectorMode vector_mode = VectorMode::RC) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_rpow,
        (APPROX, 8 /* ITERATIONS */, DST_ACCUM_MODE),
        idst,
        vector_mode,
        base_val));
}

}  // namespace ckernel
