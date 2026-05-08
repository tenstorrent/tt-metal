// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_binary_sfpu_binary_remainder.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise remainder operation with the two 32-bit integer inputs: y = remainder(x0,x1)
 * Output overwrites odst in DST.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
 * A maximum of 4 tiles from each operand can be loaded into DST at once, for a total of 8 tiles,
 * when using 16 bit formats. This gets reduced to 2 tiles from each operand for 32 bit formats.
 *
 * Return value: None
 *
 * | Argument       | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0          | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1          | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst           | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     */
// clang-format on
ALWI void remainder_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((llk_math_eltwise_binary_sfpu_remainder_int32<APPROX>(idst0, idst1, odst)));
}

/**
 * Please refer to documentation for remainder_int32_tile.
 */
ALWI void remainder_int32_tile_init() { MATH((llk_math_eltwise_binary_sfpu_remainder_int32_init<APPROX>())); }

// BF16, FP32

// clang-format off
/**
 * Performs an elementwise remainder operation with the two float inputs: y = remainder(x0,x1)
 * Computes y = x0 - x1 * floor(x0 / x1).
 * Sign correction: remainder must match the sign of x1; if (y ≠ 0) and sign(y) ≠ sign(x1) then y += x1.
 *
 * Output overwrites odst in DST.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0          | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1          | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst           | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     */
// clang-format on
ALWI void remainder_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((llk_math_eltwise_binary_sfpu_binary_remainder<APPROX, DST_ACCUM_MODE>(idst0, idst1, odst)));
}

/**
 * Please refer to documentation for remainder_binary_tile.
 */
ALWI void remainder_binary_tile_init() {
    MATH((llk_math_eltwise_binary_sfpu_binary_remainder_init<APPROX, DST_ACCUM_MODE>()));
}

}  // namespace ckernel
