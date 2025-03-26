// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_relu.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise computation of relu max (relu(max(x, upper_limit))) on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     | 
 * | upper_limit    | Upper limit of relu_min                                                    | uint32_t | Greater than 0                                        | True     |
 */
 // clang-format on
ALWI void relu_max_tile(uint32_t idst, uint32_t param0) {
    MATH((llk_math_eltwise_unary_sfpu_relu_max<APPROX>(idst, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void relu_max_tile_init() { MATH((llk_math_eltwise_unary_sfpu_relu_max_init<APPROX>())); }

// clang-format off
/**
 * Performs element-wise computation of relu min (relu(min(x, lower_limit))) on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     | 
 * | lower_limit    | Upper limit of relu_min                                                    | uint32_t | Greater than 0                                        | True     |
 */
 // clang-format on
ALWI void relu_min_tile(uint32_t idst, uint32_t param0) {
    MATH((llk_math_eltwise_unary_sfpu_relu_min<APPROX>(idst, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void relu_min_tile_init() { MATH((llk_math_eltwise_unary_sfpu_relu_min_init<APPROX>())); }

// clang-format off
/**
 * Performs element-wise computation of relu(x) = (0 if x is negative else x) on each element of a tile
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
ALWI void relu_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_relu<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void relu_tile_init() { MATH((llk_math_eltwise_unary_sfpu_relu_init<APPROX>())); }

// clang-format off
/**
 * Performs element-wise computation of leaky relu (relu(x) + slope*-relu(-x)) on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     | 
 * | slope          | slope used in leaky relu - will reinterpret unsigned int to float          | uint32_t | Greater than 0                                        | True     |
 */
 // clang-format on
ALWI void leaky_relu_tile(uint32_t idst, uint32_t slope) {
    MATH((llk_math_eltwise_unary_sfpu_lrelu<APPROX>(idst, slope)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void leaky_relu_tile_init() { MATH((llk_math_eltwise_unary_sfpu_lrelu_init<APPROX>())); }

}  // namespace ckernel
