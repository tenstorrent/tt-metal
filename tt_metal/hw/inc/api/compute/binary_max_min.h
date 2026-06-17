// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#ifdef ARCH_QUASAR
#include "llk_math_eltwise_binary_sfpu_max_min.h"
#else
#include "ckernel_sfpu_binary_max_min.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif
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
#if defined(ARCH_QUASAR)
    MATH((llk_math_eltwise_binary_sfpu_binary_max_int32<APPROX>(idst0, idst1, odst)));
#else
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binary_max_min_int32,
        (true /* IS_MAX */, false /* IS_UNSIGNED */),
        idst0,
        idst1,
        odst,
        VectorMode::RC)));
#endif
}

/**
 * Please refer to documentation.
 */
ALWI void binary_max_int32_tile_init() {
#if defined(ARCH_QUASAR)
    MATH((llk_math_eltwise_binary_sfpu_binary_max_min_int32_init()));
#else
    MATH((
        SFPU_BINARY_INIT_FN(max_int32, sfpu::binary_max_min_int32_init, (true /* IS_MAX */, false /* IS_UNSIGNED */))));
#endif
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
#ifndef ARCH_QUASAR
ALWI void binary_max_uint32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binary_max_min_int32,
        (true /* IS_MAX */, true /* IS_UNSIGNED */),
        idst0,
        idst1,
        odst,
        VectorMode::RC)));
}

/**
 * Please refer to documentation.
 */
ALWI void binary_max_uint32_tile_init() {
    MATH((
        SFPU_BINARY_INIT_FN(max_uint32, sfpu::binary_max_min_int32_init, (true /* IS_MAX */, true /* IS_UNSIGNED */))));
}
#endif

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
ALWI void binary_max_tile(uint32_t idst0, uint32_t idst1, uint32_t odst, VectorMode vector_mode = VectorMode::RC) {
#if defined(ARCH_QUASAR)
    MATH((llk_math_eltwise_binary_sfpu_binary_max<APPROX>(idst0, idst1, odst, vector_mode)));
#else
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binary_max_min,
        (true /* IS_MAX */),
        idst0,
        idst1,
        odst,
        vector_mode)));
#endif
}

/**
 * Please refer to documentation.
 */
ALWI void binary_max_tile_init() {
#if defined(ARCH_QUASAR)
    MATH((llk_math_eltwise_binary_sfpu_binary_max_min_init()));
#else
    MATH((SFPU_BINARY_INIT_FN(max, sfpu::binary_max_min_init, (true /* IS_MAX */))));
#endif
}

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
#if defined(ARCH_QUASAR)
    MATH((llk_math_eltwise_binary_sfpu_binary_min_int32<APPROX>(idst0, idst1, odst)));
#else
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binary_max_min_int32,
        (false /* IS_MAX */, false /* IS_UNSIGNED */),
        idst0,
        idst1,
        odst,
        VectorMode::RC)));
#endif
}

/**
 * Please refer to documentation.
 */
ALWI void binary_min_int32_tile_init() {
#if defined(ARCH_QUASAR)
    MATH((llk_math_eltwise_binary_sfpu_binary_max_min_int32_init()));
#else
    MATH((SFPU_BINARY_INIT_FN(
        min_int32, sfpu::binary_max_min_int32_init, (false /* IS_MAX */, false /* IS_UNSIGNED */))));
#endif
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
#ifndef ARCH_QUASAR
ALWI void binary_min_uint32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binary_max_min_int32,
        (false /* IS_MAX */, true /* IS_UNSIGNED */),
        idst0,
        idst1,
        odst,
        VectorMode::RC)));
}

/**
 * Please refer to documentation.
 */
ALWI void binary_min_uint32_tile_init() {
    MATH((SFPU_BINARY_INIT_FN(
        min_uint32, sfpu::binary_max_min_int32_init, (false /* IS_MAX */, true /* IS_UNSIGNED */))));
}
#endif

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
ALWI void binary_min_tile(uint32_t idst0, uint32_t idst1, uint32_t odst, VectorMode vector_mode = VectorMode::RC) {
#if defined(ARCH_QUASAR)
    MATH((llk_math_eltwise_binary_sfpu_binary_min<APPROX>(idst0, idst1, odst, vector_mode)));
#else
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binary_max_min,
        (false /* IS_MAX */),
        idst0,
        idst1,
        odst,
        vector_mode)));
#endif
}

/**
 * Please refer to documentation.
 */
ALWI void binary_min_tile_init() {
#if defined(ARCH_QUASAR)
    MATH((llk_math_eltwise_binary_sfpu_binary_max_min_init()));
#else
    MATH((SFPU_BINARY_INIT_FN(min, sfpu::binary_max_min_init, (false /* IS_MAX */))));
#endif
}

}  // namespace ckernel
