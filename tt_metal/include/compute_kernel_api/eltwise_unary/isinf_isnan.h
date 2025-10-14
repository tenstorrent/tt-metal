// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {
// clang-format off
/**
 * Will store in the output of the compute core True if the input tile is infinity.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void isinf_tile(uint32_t idst) {
    MATH(SFPU_UNARY_NO_PARAM_KERNEL_WITH_TYPE_AND_ITERATIONS(_calculate_sfpu_isinf_isnan_, isinf, RC, APPROX, idst, 8));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void isinf_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(isinf, APPROX)); }

// clang-format off
/**
 * Will store in the output of the compute core True if the input tile is positive infinity.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void isposinf_tile(uint32_t idst) {
    MATH(SFPU_UNARY_NO_PARAM_KERNEL_WITH_TYPE_AND_ITERATIONS(_calculate_sfpu_isinf_isnan_, isposinf, RC, APPROX, idst, 8));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void isposinf_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(isposinf, APPROX)); }

// clang-format off
/**
 * Will store in the output of the compute core True if the input tile is negative infinity.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void isneginf_tile(uint32_t idst) {
    MATH(SFPU_UNARY_NO_PARAM_KERNEL_WITH_TYPE_AND_ITERATIONS(_calculate_sfpu_isinf_isnan_, isneginf, RC, APPROX, idst, 8));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void isneginf_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(isneginf, APPROX)); }

// clang-format off
/**
 * Will store in the output of the compute core True if the input tile is nan.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void isnan_tile(uint32_t idst) {
    MATH(SFPU_UNARY_NO_PARAM_KERNEL_WITH_TYPE_AND_ITERATIONS(_calculate_sfpu_isinf_isnan_, isnan, RC, APPROX, idst, 8));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void isnan_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(isnan, APPROX)); }

// clang-format off
/**
 * Will store in the output of the compute core True if the input tile is finite
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void isfinite_tile(uint32_t idst) {
    MATH(SFPU_UNARY_NO_PARAM_KERNEL_WITH_TYPE_AND_ITERATIONS(_calculate_sfpu_isinf_isnan_, isfinite, RC, APPROX, idst, 8));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void isfinite_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(isfinite, APPROX)); }
}  // namespace ckernel
