// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_rrelu.h"
#endif

namespace ckernel {

/**
 * Performs element-wise RReLU (Randomized Leaky ReLU) in evaluation mode.
 * RReLU_eval(x) = x if x >= 0, slope * x if x < 0
 * where slope = (lower + upper) / 2, pre-computed by host.
 *
 * | Argument   | Description                                                                | Type     | Valid Range                                           | Required |
 * |------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst       | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | slope_bits | Bit-cast uint32 representation of the slope float value                    | uint32_t |                                                       | True     |
 */
ALWI void rrelu_eval_tile(uint32_t idst, uint32_t slope_bits) {
    MATH((llk_math_eltwise_unary_sfpu_rrelu_eval<APPROX>(idst, (int)VectorMode::RC, slope_bits)));
}

/**
 * Performs element-wise RReLU (Randomized Leaky ReLU) in training mode.
 * RReLU_train(x) = x if x >= 0, a * x if x < 0
 * where a ~ Uniform(lower, upper) per element.
 *
 * | Argument    | Description                                                                | Type     | Valid Range                                           | Required |
 * |-------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst        | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | lower_bits  | Bit-cast uint32 representation of the lower bound float value              | uint32_t |                                                       | True     |
 * | upper_bits  | Bit-cast uint32 representation of the upper bound float value              | uint32_t |                                                       | True     |
 */
ALWI void rrelu_train_tile(uint32_t idst, uint32_t lower_bits, uint32_t upper_bits) {
    MATH((llk_math_eltwise_unary_sfpu_rrelu_train<APPROX>(idst, (int)VectorMode::RC, lower_bits, upper_bits)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void rrelu_tile_init() { MATH((llk_math_eltwise_unary_sfpu_rrelu_init<APPROX>())); }

}  // namespace ckernel
