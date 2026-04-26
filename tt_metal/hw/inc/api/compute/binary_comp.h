// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_binary_comp.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise comparison operation with the two integer inputs: y = comparison_op(x0,x1)
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
 * | odst           | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void lt_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binary_comp_int32,
        (APPROX, 8, SfpuType::lt),
        idst0,
        idst1,
        odst,
        (int)VectorMode::RC)));
}

ALWI void gt_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binary_comp_int32,
        (APPROX, 8, SfpuType::gt),
        idst0,
        idst1,
        odst,
        (int)VectorMode::RC)));
}

ALWI void ge_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binary_comp_int32,
        (APPROX, 8, SfpuType::ge),
        idst0,
        idst1,
        odst,
        (int)VectorMode::RC)));
}

ALWI void le_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binary_comp_int32,
        (APPROX, 8, SfpuType::le),
        idst0,
        idst1,
        odst,
        (int)VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void lt_int32_tile_init() { MATH((SFPU_BINARY_INIT(lt))); }

ALWI void gt_int32_tile_init() { MATH((SFPU_BINARY_INIT(gt))); }

ALWI void ge_int32_tile_init() { MATH((SFPU_BINARY_INIT(ge))); }

ALWI void le_int32_tile_init() { MATH((SFPU_BINARY_INIT(le))); }

// clang-format off
/**
 * Performs an elementwise comparison operation with the two uint16 inputs: y = comparison_op(x0,x1)
 * Output overwrites odst in DST.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
 * A maximum of 4 tiles from each operand can be loaded into DST at once, for a total of 8 tiles,
 * when using 16 bit formats.
 *
 * Return value: None
 *
 * | Argument       | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0          | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1          | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst           | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on

ALWI void lt_uint16_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binary_comp_uint16,
        (APPROX, 8, SfpuType::lt),
        idst0,
        idst1,
        odst,
        (int)VectorMode::RC)));
}

ALWI void gt_uint16_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binary_comp_uint16,
        (APPROX, 8, SfpuType::gt),
        idst0,
        idst1,
        odst,
        (int)VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */

ALWI void lt_uint16_tile_init() { MATH((SFPU_BINARY_INIT(lt))); }

ALWI void gt_uint16_tile_init() { MATH((SFPU_BINARY_INIT(gt))); }

}  // namespace ckernel
