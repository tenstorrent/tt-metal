// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
 * Performs element-wise RReLU operation (eval mode): x if x >= 0, slope * x if x < 0
 * where slope = (lower + upper) / 2.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The slope value (bit-cast float to uint32_t)                               | uint32_t | Any valid float bit pattern                           | True     |
 */
// clang-format on
ALWI void rrelu_tile(uint32_t idst, uint32_t param0) {
    MATH((llk_math_eltwise_unary_sfpu_rrelu<APPROX>(idst, (int)VectorMode::RC, param0)));
}

/**
 * Performs element-wise RReLU operation (training mode): x if x >= 0, a * x if x < 0
 * where a is sampled from Uniform(lower, upper) per element.
 *
 * | Argument        | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be
 * less than the size of the DST register buffer | True     | | param0          | The lower bound (bit-cast float to
 * uint32_t)                               | uint32_t | Any valid float bit pattern                           | True |
 * | param1          | The upper bound (bit-cast float to uint32_t)                               | uint32_t | Any valid
 * float bit pattern                           | True     |
 */
ALWI void rrelu_tile(uint32_t idst, uint32_t param0, uint32_t param1) {
    MATH((llk_math_eltwise_unary_sfpu_rrelu_training<APPROX>(idst, (int)VectorMode::RC, param0, param1)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void rrelu_tile_init() { MATH((llk_math_eltwise_unary_sfpu_rrelu_init<APPROX>())); }

/**
 * Init for training mode (seeds PRNG).
 */
ALWI void rrelu_tile_init(uint32_t seed) { MATH((llk_math_eltwise_unary_sfpu_rrelu_training_init<APPROX>(seed))); }

}  // namespace ckernel
