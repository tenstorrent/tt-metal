// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_binary_max_min.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise maximum operation on inputs of int32 data type at idst0, idst1: y = max(x0, x1).
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
ALWI void binary_max_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binary_max_min_int32,
        (true, false),
        idst0,
        idst1,
        odst,
        (int)VectorMode::RC)));
}

/**
 * Please refer to documentation.
 */
ALWI void binary_max_int32_tile_init() {
    MATH((SFPU_BINARY_INIT_CB(max_int32, sfpu::binary_max_min_int32_init, (true, false))));
}

// clang-format off
/**
 * Performs an elementwise maximum operation on inputs of uint32 data type at idst0, idst1: y = max(x0, x1).
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
ALWI void binary_max_uint32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binary_max_min_int32,
        (true, true),
        idst0,
        idst1,
        odst,
        (int)VectorMode::RC)));
}

/**
 * Please refer to documentation.
 */
ALWI void binary_max_uint32_tile_init() {
    MATH((SFPU_BINARY_INIT_CB(max_uint32, sfpu::binary_max_min_int32_init, (true, true))));
}

// clang-format off
/**
 * Performs an elementwise maximum operation on inputs at idst0, idst1: y = max(x0, x1).
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
ALWI void binary_max_tile(uint32_t idst0, uint32_t idst1, uint32_t odst, int vector_mode = (int)VectorMode::RC) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_binary_max_min, (true), idst0, idst1, odst, vector_mode)));
}

/**
 * Please refer to documentation.
 */
ALWI void binary_max_tile_init() { MATH((SFPU_BINARY_INIT_CB(max, sfpu::binary_max_min_init, (true)))); }

// clang-format off
/**
 * Performs an elementwise minimum operation on inputs of int32 data type at idst0, idst1: y = min(x0, x1).
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
ALWI void binary_min_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binary_max_min_int32,
        (false, false),
        idst0,
        idst1,
        odst,
        (int)VectorMode::RC)));
}

/**
 * Please refer to documentation.
 */
ALWI void binary_min_int32_tile_init() {
    MATH((SFPU_BINARY_INIT_CB(min_int32, sfpu::binary_max_min_int32_init, (false, false))));
}

// clang-format off
/**
 * Performs an elementwise minimum operation on inputs of uint32 data type at idst0, idst1: y = min(x0, x1).
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
ALWI void binary_min_uint32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binary_max_min_int32,
        (false, true),
        idst0,
        idst1,
        odst,
        (int)VectorMode::RC)));
}

/**
 * Please refer to documentation.
 */
ALWI void binary_min_uint32_tile_init() {
    MATH((SFPU_BINARY_INIT_CB(min_uint32, sfpu::binary_max_min_int32_init, (false, true))));
}

// clang-format off
/**
 * Performs an elementwise minimum operation on inputs at idst0, idst1: y = min(x0, x1).
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
ALWI void binary_min_tile(uint32_t idst0, uint32_t idst1, uint32_t odst, int vector_mode = (int)VectorMode::RC) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_binary_max_min, (false), idst0, idst1, odst, vector_mode)));
}

/**
 * Please refer to documentation.
 */
ALWI void binary_min_tile_init() { MATH((SFPU_BINARY_INIT_CB(min, sfpu::binary_max_min_init, (false)))); }

}  // namespace ckernel
