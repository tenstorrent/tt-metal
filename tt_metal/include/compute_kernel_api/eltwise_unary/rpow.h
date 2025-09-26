// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#include "llk_math_eltwise_unary_sfpu_rpow.h"

namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */
ALWI void rpow_tile_init() { MATH((llk_math_eltwise_unary_sfpu_rpow_init<APPROX>())); }
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
ALWI void rpow_tile(uint32_t idst, uint32_t base_val, int vector_mode = (int)VectorMode::RC) {
    MATH((llk_math_eltwise_unary_sfpu_rpow<APPROX, DST_ACCUM_MODE>(idst, base_val, vector_mode)));
}

}  // namespace ckernel
