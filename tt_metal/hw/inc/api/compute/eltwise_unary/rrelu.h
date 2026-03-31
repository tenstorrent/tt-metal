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
 * Performs element-wise RReLU (Randomized Leaky ReLU) operation.
 * In evaluation mode (seed=0): output = x if x >= 0, output = ((lower + upper) / 2) * x if x < 0
 * In training mode (seed!=0): output = x if x >= 0, output = a * x if x < 0 where a ~ U(lower, upper)
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | Lower bound of the uniform distribution (bit-cast float)                   | uint32_t |                                                       | True     |
 * | param1          | Upper bound of the uniform distribution (bit-cast float)                   | uint32_t |                                                       | True     |
 * | param2          | PRNG seed (0 = eval mode, non-zero = training mode with seed)              | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void rrelu_tile(uint32_t idst, uint32_t param0, uint32_t param1, uint32_t param2) {
    MATH((llk_math_eltwise_unary_sfpu_rrelu<APPROX>(idst, param0, param1, param2)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void rrelu_tile_init(uint32_t seed) { MATH((llk_math_eltwise_unary_sfpu_rrelu_init<APPROX>(seed))); }

}  // namespace ckernel
