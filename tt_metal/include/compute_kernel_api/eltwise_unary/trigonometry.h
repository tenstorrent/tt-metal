// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_trigonometry.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */
ALWI void sin_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(sine, APPROX)); }

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
ALWI void sin_tile(uint32_t idst) { MATH(SFPU_UNARY_NO_PARAM_KERNEL_WITH_TYPE(sfpu_trig, sine, RC, APPROX, idst)); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void cos_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(cosine, APPROX)); }

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
ALWI void cos_tile(uint32_t idst) { MATH(SFPU_UNARY_NO_PARAM_KERNEL_WITH_TYPE(sfpu_trig, cosine, RC, APPROX, idst)); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void acosh_tile_init() { MATH(SFPU_INIT_KERNEL_CALL(acosh, ckernel::sfpu::_init_inverse_hyperbolic_, APPROX)); }

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
ALWI void acosh_tile(uint32_t idst) {
    MATH(SFPU_TWO_PARAM_KERNEL(_calculate_acosh_, APPROX, 8, idst, (int)VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void tan_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(tan, APPROX)); }

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
ALWI void tan_tile(uint32_t idst) { MATH(SFPU_UNARY_NO_PARAM_KERNEL_WITH_TYPE(sfpu_trig, tan, RC, APPROX, idst)); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void asinh_tile_init() { MATH(SFPU_INIT_KERNEL_CALL(asinh, ckernel::sfpu::_init_inverse_hyperbolic_, APPROX)); }

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
ALWI void asinh_tile(uint32_t idst) {
    MATH(SFPU_TWO_PARAM_KERNEL(_calculate_asinh_, APPROX, 8, idst, (int)VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void atanh_tile_init() { MATH(SFPU_INIT_KERNEL_CALL(atanh, ckernel::sfpu::_init_atanh_, APPROX)); }

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
ALWI void atanh_tile(uint32_t idst) {
    MATH(
        SFPU_THREE_PARAM_KERNEL_FP32_FIRST(_calculate_atanh_, APPROX, DST_ACCUM_MODE, 8, idst, (int)VectorMode::RC));
}

// clang-format off
/**
 * Performs element-wise computation of arcsine on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void asin_tile(uint32_t idst) { MATH(SFPU_UNARY_NO_PARAM_KERNEL(asin, RC, true, idst)); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void asin_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(asin, true)); }

// clang-format off
/**
 * Performs element-wise computation of arctan on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void atan_tile(uint32_t idst) {
    MATH(SFPU_THREE_PARAM_KERNEL_FP32_FIRST(calculate_atan, APPROX, DST_ACCUM_MODE, 8, idst, (int)VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void atan_tile_init() { MATH(SFPU_INIT_KERNEL_CALL(atan, sfpu::atan_init, true)); }

// clang-format off
/**
 * Performs element-wise computation of arcossine on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void acos_tile(uint32_t idst) { MATH(SFPU_UNARY_NO_PARAM_KERNEL(acos, RC, true, idst)); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void acos_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(acos, true)); }

/**
* Please refer to documentation for any_init.
*/
ALWI void cosh_tile_init() { MATH(SFPU_INIT_KERNEL_CALL(cosh, ckernel::sfpu::init_hyperbolic_trig, APPROX)); }

// clang-format off
/**
 * Performs element-wise computation of the trigonometric hyperbolic cosine operation on each element of a tile
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
ALWI void cosh_tile(uint32_t idst) {
    MATH(
        SFPU_THREE_PARAM_KERNEL_FP32_FIRST(calculate_cosh, APPROX, DST_ACCUM_MODE, 8, idst, (int)VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void sinh_tile_init() { MATH(SFPU_INIT_KERNEL_CALL(sinh, ckernel::sfpu::init_hyperbolic_trig, APPROX)); }

// clang-format off
/**
 * Performs element-wise computation of the trigonometric hyperbolic sine operation on each element of a tile
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
ALWI void sinh_tile(uint32_t idst) {
    MATH(
        SFPU_THREE_PARAM_KERNEL_FP32_FIRST(calculate_sinh, APPROX, DST_ACCUM_MODE, 8, idst, (int)VectorMode::RC));
}

}  // namespace ckernel
