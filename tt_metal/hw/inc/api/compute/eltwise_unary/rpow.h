// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_rpow.h"
#endif

namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */
ALWI void rpow_tile_init() { MATH((llk_math_eltwise_unary_sfpu_rpow_init<APPROX>())); }
// clang-format off
/**
 * Performs element-wise computation of the rpow on each element of a tile
 * where rpow(x, base) = pow(base, x) = base^x
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                 | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the computation on  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | base_val       | The base value as IEEE 754 float bits to raise to the power of each element | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void rpow_tile(uint32_t idst, uint32_t base_val) {
    MATH((llk_math_eltwise_unary_sfpu_rpow<APPROX>(idst, base_val)));
}

}  // namespace ckernel
