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
 * Performs element-wise Randomized Leaky ReLU (RReLU):
 *   output = x           if x >= 0
 *   output = a * x       if x < 0
 *
 * In eval mode: a = (lower + upper) / 2.
 * In training mode: a ~ Uniform(lower, upper) per element.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param_lower     | Bit-cast float lower bound of the uniform distribution                     | uint32_t | Any valid float bit pattern                           | True     |
 * | param_upper     | Bit-cast float upper bound of the uniform distribution                     | uint32_t | Any valid float bit pattern                           | True     |
 * | param_training  | Bit-cast float training flag (0.0f = eval, 1.0f = training)                | uint32_t | 0x00000000 or 0x3F800000                              | True     |
 */
// clang-format on
ALWI void rrelu_tile(uint32_t idst, uint32_t param_lower, uint32_t param_upper, uint32_t param_training) {
    MATH((llk_math_eltwise_unary_sfpu_rrelu<APPROX>(idst, param_lower, param_upper, param_training)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void rrelu_tile_init() { MATH((llk_math_eltwise_unary_sfpu_rrelu_init<APPROX>())); }

}  // namespace ckernel
