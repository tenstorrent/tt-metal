// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_trigonometry.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */
ALWI void sin_tile_init() { MATH((llk_math_eltwise_unary_sfpu_sine_init<APPROX>())); }

// clang-format off
/**
 * Performs element-wise computation of the trigonometric sine operation on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
 // clang-format on
ALWI void sin_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_sine_op<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void cos_tile_init() { MATH((llk_math_eltwise_unary_sfpu_cosine_init<APPROX>())); }

// clang-format off
/**
 * Performs element-wise computation of the trigonometric cosine operation on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
 // clang-format on
ALWI void cos_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_cosine_op<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void acosh_tile_init() { MATH((llk_math_eltwise_unary_sfpu_acosh_init<APPROX>())); }

// clang-format off
/**
 * Performs element-wise computation of the inverse hyperbolic cosine operation on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void acosh_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_acosh<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void tan_tile_init() { MATH((llk_math_eltwise_unary_sfpu_tan_init<APPROX>())); }

// clang-format off
/**
 * Performs element-wise computation of the trigonometric tan operation on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
 // clang-format on
ALWI void tan_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_tan_op<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void asinh_tile_init() { MATH((llk_math_eltwise_unary_sfpu_asinh_init<APPROX>())); }

// clang-format off
/**
 * Performs element-wise computation of the inverse hyperbolic sine operation on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void asinh_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_asinh<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void atanh_tile_init() { MATH((llk_math_eltwise_unary_sfpu_atanh_init<APPROX>())); }

// clang-format off
 /**
  * Performs element-wise computation of the inverse hyperbolic tangent operation on each element of a tile
  * in DST register at index tile_index. The DST register buffer must be in
  * acquired state via *acquire_dst* call. This call is blocking and is only
  * available on the compute engine.
  *
  * Return value: None
  *
  * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
  * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
  * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
  */
// clang-format on
ALWI void atanh_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_atanh<APPROX, DST_ACCUM_MODE>(idst))); }

}  // namespace ckernel
