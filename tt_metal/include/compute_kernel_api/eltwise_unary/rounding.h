// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_rounding_ops.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */
ALWI void rounding_op_tile_init() { MATH((llk_math_eltwise_unary_sfpu_rounding_op_init<APPROX>())); }

// clang-format off
/**
 * Performs ceil operation on each row of a tile.
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform ceil operation     | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void ceil_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_ceil<APPROX>(idst))); }

// clang-format off
/**
 * Performs ceil operation on each row of a tile.
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform ceil operation     | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void ceil_tile_float32(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_ceil_float32<APPROX>(idst))); }

// clang-format off
/**
 * Performs floor operation on each row of a tile.
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform floor operation    | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void floor_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_floor<APPROX>(idst))); }

// clang-format off
/**
 * Performs floor operation on each row of a tile.
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform floor operation    | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void floor_tile_float32(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_floor_float32<APPROX>(idst))); }

// clang-format off
/**
 * Performs trunc operation on each row of a tile.
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform trunc operation    | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void trunc_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_trunc<APPROX>(idst))); }

// clang-format off
/**
 * Performs trunc operation on each row of a tile.
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform trunc operation    | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void trunc_tile_float32(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_trunc_float32<APPROX>(idst))); }

// clang-format off
/**
 * Performs element-wise computation of the round operation on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | decimals        | The number of decimal places to round to.                                  | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void round_tile(uint32_t idst, int32_t decimals) {
    MATH((llk_math_eltwise_unary_sfpu_round<APPROX>(idst, decimals)));
}

}  // namespace ckernel
