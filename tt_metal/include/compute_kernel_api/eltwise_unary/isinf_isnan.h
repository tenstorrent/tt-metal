// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_isinf_isnan.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
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
ALWI void isinf_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_params<APPROX>(
        ckernel::sfpu::calculate_sfpu_isinf_isnan<SfpuType::isinf, APPROX>, idst, (int)VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void isinf_tile_init() { MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::isinf, APPROX>())); }

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
    MATH((llk_math_eltwise_unary_sfpu_params<APPROX>(
        ckernel::sfpu::calculate_sfpu_isinf_isnan<SfpuType::isposinf, APPROX>, idst, (int)VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void isposinf_tile_init() { MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::isposinf, APPROX>())); }

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
    MATH((llk_math_eltwise_unary_sfpu_params<APPROX>(
        ckernel::sfpu::calculate_sfpu_isinf_isnan<SfpuType::isneginf, APPROX>, idst, (int)VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void isneginf_tile_init() { MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::isneginf, APPROX>())); }

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
    MATH((llk_math_eltwise_unary_sfpu_params<APPROX>(
        ckernel::sfpu::calculate_sfpu_isinf_isnan<SfpuType::isnan, APPROX>, idst, (int)VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void isnan_tile_init() { MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::isnan, APPROX>())); }

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
    MATH((llk_math_eltwise_unary_sfpu_params<APPROX>(
        ckernel::sfpu::calculate_sfpu_isinf_isnan<SfpuType::isfinite, APPROX>, idst, (int)VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void isfinite_tile_init() { MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::isfinite, APPROX>())); }
}  // namespace ckernel
