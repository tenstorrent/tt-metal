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
 * In evaluation mode: output = x if x >= 0, slope * x if x < 0
 * where slope = (lower + upper) / 2.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | slope           | The negative slope as a bfloat16-encoded uint32_t                          | uint32_t | Any valid bfloat16 value                              | True     |
 */
// clang-format on
ALWI void rrelu_tile(uint32_t idst, uint32_t slope) {
    MATH((llk_math_eltwise_unary_sfpu_rrelu<APPROX>(idst, (int)VectorMode::RC, slope)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void rrelu_tile_init() { MATH((llk_math_eltwise_unary_sfpu_rrelu_init<APPROX>())); }

}  // namespace ckernel
