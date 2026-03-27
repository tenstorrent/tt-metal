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
 * Performs element-wise RReLU (Randomized Leaky ReLU) in eval/inference mode.
 *   RReLU(x) = x              if x >= 0
 *   RReLU(x) = slope * x      if x < 0
 *   where slope = (lower + upper) / 2
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | Lower bound of the uniform distribution (bitcast float)                    | uint32_t |                                                       | True     |
 * | param1          | Upper bound of the uniform distribution (bitcast float)                    | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void rrelu_tile(uint32_t idst, uint32_t param0, uint32_t param1) {
    MATH((llk_math_eltwise_unary_sfpu_rrelu<APPROX, DST_ACCUM_MODE>(idst, param0, param1)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void rrelu_tile_init() { MATH((llk_math_eltwise_unary_sfpu_rrelu_init<APPROX>())); }

// clang-format off
/**
 * Performs element-wise RReLU (Randomized Leaky ReLU) in training mode.
 *   RReLU(x) = x              if x >= 0
 *   RReLU(x) = a * x          if x < 0
 *   where a ~ Uniform(lower, upper) is sampled per element using hardware PRNG.
 *
 * PRNG must be initialized via rrelu_tile_training_init(seed) before calling this function.
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | Lower bound of the uniform distribution (bitcast float)                    | uint32_t |                                                       | True     |
 * | param1          | Range = upper - lower (bitcast float)                                      | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void rrelu_tile_training(uint32_t idst, uint32_t param0, uint32_t param1) {
    MATH((llk_math_eltwise_unary_sfpu_rrelu_training<APPROX, DST_ACCUM_MODE>(idst, param0, param1)));
}

/**
 * Initialize PRNG for RReLU training mode. Must be called once before rrelu_tile_training.
 */
ALWI void rrelu_tile_training_init(uint32_t seed) {
    MATH((llk_math_eltwise_unary_sfpu_rrelu_training_init<APPROX>(seed)));
}

}  // namespace ckernel
