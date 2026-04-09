// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
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
 * Performs element-wise Randomized Leaky ReLU (RReLU).
 *   f(x) = x          when x >= 0
 *   f(x) = a * x      when x < 0
 * where a = (lower + upper) / 2 in eval mode, or a ~ Uniform(lower, upper) in training mode.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | training        | 0 for eval mode, 1 for training mode                                      | uint32_t | 0 or 1                                                | True     |
 */
// clang-format on
ALWI void rrelu_tile(uint32_t idst, uint32_t training) {
    MATH((llk_math_eltwise_unary_sfpu_rrelu<APPROX>(idst, (int)VectorMode::RC, training)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void rrelu_tile_init(uint32_t lower_bits, uint32_t upper_bits) {
    MATH((llk_math_eltwise_unary_sfpu_rrelu_init<APPROX>(lower_bits, upper_bits)));
}

}  // namespace ckernel
