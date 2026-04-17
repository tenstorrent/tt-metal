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
 * Performs element-wise Randomized Leaky ReLU (RReLU) operation.
 *   f(x) = x          when x >= 0
 *   f(x) = a * x      when x < 0
 * In eval mode a = (lower + upper) / 2; in train mode a ~ Uniform(lower, upper).
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | lower bound (bit-cast float as uint32)                                     | uint32_t |                                                       | True     |
 * | param1          | upper bound (bit-cast float as uint32)                                     | uint32_t |                                                       | True     |
 * | param2          | training flag (0 = eval, nonzero = train)                                  | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void rrelu_tile(uint32_t idst, uint32_t param0, uint32_t param1, uint32_t param2) {
    MATH((llk_math_eltwise_unary_sfpu_rrelu<APPROX>(idst, param0, param1, param2)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void rrelu_tile_init() { MATH((llk_math_eltwise_unary_sfpu_rrelu_init<APPROX>())); }

}  // namespace ckernel
