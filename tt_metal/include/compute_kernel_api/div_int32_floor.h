// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_binary_sfpu_div_int32_floor.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise division operation with the two 32-bit integer inputs: y = divide(x0,x1)
 * Output overwrites odst in DST.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
 * A maximum of 2 tiles from each operand can be loaded into DST at once, for a total of 4 tiles.
 *
 * Return value: None
 *
 * | Argument              | Description                                                           | Type     | Valid Range                                           | Required |
 * |-----------------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0                 | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1                 | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst                  | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void div_int32_floor_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((llk_math_eltwise_binary_sfpu_div_int32_floor<APPROX>(idst0, idst1, odst)));
}
ALWI void div_int32_trunc_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((llk_math_eltwise_binary_sfpu_div_int32_trunc<APPROX>(idst0, idst1, odst)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void div_int32_floor_tile_init() { MATH((llk_math_eltwise_binary_sfpu_div_int32_floor_init<APPROX>())); }
ALWI void div_int32_trunc_tile_init() { MATH((llk_math_eltwise_binary_sfpu_div_int32_trunc_init<APPROX>())); }

}  // namespace ckernel
