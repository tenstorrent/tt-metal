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
 * For x >= 0: f(x) = x
 * For x < 0:  f(x) = a * x
 *   eval mode:     a = (lower + upper) / 2
 *   training mode: a ~ Uniform(lower, upper) per element
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | lower           | Lower bound of uniform distribution (bit-cast float)                       | uint32_t |                                                       | True     |
 * | range           | upper - lower (bit-cast float)                                             | uint32_t |                                                       | True     |
 * | seed            | 0 for eval, nonzero PRNG seed for training                                 | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void rrelu_tile(uint32_t idst, uint32_t lower, uint32_t range, uint32_t seed) {
    MATH((llk_math_eltwise_unary_sfpu_rrelu<APPROX>(idst, lower, range, seed)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void rrelu_tile_init(uint32_t seed = 0) { MATH((llk_math_eltwise_unary_sfpu_rrelu_init<APPROX>(seed))); }

}  // namespace ckernel
