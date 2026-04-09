// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_rrelu.h"
#endif

namespace ckernel {

// clang-format off
 /**
 * Performs element-wise RReLU (eval mode): x if x >= 0, slope * x if x < 0.
 * slope = (lower + upper) / 2, precomputed on host and passed as bit-cast uint32.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | slope_u32       | Bit-cast uint32 of the precomputed eval slope                              | uint32_t | Any valid float bit pattern                           | True     |
 */
// clang-format on
ALWI void rrelu_eval_tile(uint32_t idst, uint32_t slope_u32) {
    MATH((llk_math_eltwise_unary_sfpu_rrelu_eval<APPROX>(idst, slope_u32)));
}

// clang-format off
 /**
 * Performs element-wise RReLU (training mode): x if x >= 0, a * x if x < 0
 * where a ~ Uniform(lower, upper) per element.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | lower_u32       | Bit-cast uint32 of the lower bound                                         | uint32_t | Any valid float bit pattern                           | True     |
 * | upper_u32       | Bit-cast uint32 of the upper bound                                         | uint32_t | Any valid float bit pattern                           | True     |
 */
// clang-format on
ALWI void rrelu_train_tile(uint32_t idst, uint32_t lower_u32, uint32_t upper_u32) {
    MATH((llk_math_eltwise_unary_sfpu_rrelu_train<APPROX>(idst, lower_u32, upper_u32)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void rrelu_tile_init() { MATH((llk_math_eltwise_unary_sfpu_rrelu_init<APPROX>())); }

}  // namespace ckernel
