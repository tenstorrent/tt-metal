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
 * Performs element-wise RReLU (Randomized Leaky ReLU) operation on a tile in DST register at index idst.
 * RReLU(x) = x if x >= 0, a*x if x < 0
 * In eval mode: a = (lower + upper) / 2 (deterministic)
 * In training mode: a ~ Uniform(lower, upper) (random per element)
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | Lower bound (bitcast float)                                               | uint32_t |                                                       | True     |
 * | param1          | Range = upper - lower (bitcast float)                                     | uint32_t |                                                       | True     |
 * | param2          | Training mode flag (bitcast float: 1.0 = training, 0.0 = eval)           | uint32_t |                                                       | True     |
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
