// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "ckernel_sfpu_relu.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

ALWI void relu_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(relu_min, APPROX)); }

// clang-format off
/**
 * Performs element-wise computation of relu(x) = (0 if x is negative else x) on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *tile_regs_acquire* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void relu_tile(uint32_t idst) { MATH(SFPU_UNARY_ONE_PARAM_KERNEL_FN_FLOAT(_relu_min_, RC, APPROX, idst, 0)); }
#ifndef ARCH_QUASAR
// clang-format off
/**
 * Performs element-wise computation of relu max (relu(max(x, upper_limit))) on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *tile_regs_acquire* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | upper_limit    | Upper limit of relu_min                                                    | uint32_t | Greater than 0                                        | True     |
 */
// clang-format on

ALWI void relu_max_tile(uint32_t idst, uint32_t param0) {
    MATH(SFPU_UNARY_ONE_PARAM_KERNEL_FN_FLOAT(_relu_max_, RC, APPROX, idst, param0));
}
ALWI void relu_max_tile_pack(uint32_t idst, uint32_t param0) {
    PACK(SFPU_UNARY_ONE_PARAM_KERNEL_FN_FLOAT(_relu_max_, RC, APPROX, idst, param0));
}

ALWI void relu_max_tile_int32(uint32_t idst, uint32_t param0) {
    MATH(SFPU_UNARY_ONE_PARAM_KERNEL_FN_INT(_relu_max_, RC, APPROX, idst, param0));
}

ALWI void relu_max_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(relu_max, APPROX)); }
ALWI void relu_max_tile_init_pack() { PACK(SFPU_UNARY_KERNEL_INIT(relu_max, APPROX)); }

// clang-format off
/**
 * Performs element-wise computation of relu min (relu(min(x, lower_limit))) on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *tile_regs_acquire* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | lower_limit    | Upper limit of relu_min                                                    | uint32_t | Greater than 0                                        | True     |
 */
// clang-format on
ALWI void relu_min_tile(uint32_t idst, uint32_t param0) {
    MATH(SFPU_UNARY_ONE_PARAM_KERNEL_FN_FLOAT(_relu_min_, RC, APPROX, idst, param0));
}

ALWI void relu_min_tile_int32(uint32_t idst, uint32_t param0) {
    MATH(SFPU_UNARY_ONE_PARAM_KERNEL_FN_INT(_relu_min_, RC, APPROX, idst, param0));
}

ALWI void relu_min_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(relu_min, APPROX)); }

ALWI void relu_tile_int32(uint32_t idst) { MATH(SFPU_UNARY_ONE_PARAM_KERNEL_FN_INT(_relu_min_, RC, APPROX, idst, 0)); }

// clang-format off
/**
 * Performs element-wise computation of leaky relu (relu(x) + slope*-relu(-x)) on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *tile_regs_acquire* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | slope          | slope used in leaky relu - will reinterpret unsigned int to float          | uint32_t | Greater than 0                                        | True     |
 */
// clang-format on
ALWI void leaky_relu_tile(uint32_t idst, uint32_t slope = 0) {
    MATH(SFPU_UNARY_ONE_PARAM_KERNEL_FN(calculate_lrelu, RC, APPROX, idst, slope));
}

ALWI void leaky_relu_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(lrelu, APPROX)); }
#endif
}  // namespace ckernel
