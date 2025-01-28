// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_isinf_isnan.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
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
ALWI void isinf_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_isinf<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void isinf_tile_init() { MATH((llk_math_eltwise_unary_sfpu_isinf_init<APPROX>())); }

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
ALWI void isposinf_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_isposinf<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void isposinf_tile_init() { MATH((llk_math_eltwise_unary_sfpu_isposinf_init<APPROX>())); }

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
ALWI void isneginf_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_isneginf<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void isneginf_tile_init() { MATH((llk_math_eltwise_unary_sfpu_isneginf_init<APPROX>())); }

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
ALWI void isnan_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_isnan<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void isnan_tile_init() { MATH((llk_math_eltwise_unary_sfpu_isnan_init<APPROX>())); }

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
ALWI void isfinite_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_isfinite<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void isfinite_tile_init() { MATH((llk_math_eltwise_unary_sfpu_isfinite_init<APPROX>())); }
}  // namespace ckernel
