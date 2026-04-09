// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_rrelu.h"
#endif

namespace ckernel {

/**
 * Performs element-wise RReLU operation.
 * RReLU(x) = x if x >= 0, a * x if x < 0
 * where a is determined by lower and upper bounds (passed as bit-cast uint32_t floats).
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * | Argument | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | lower    | Lower bound of the uniform distribution (bit-cast float)                   | uint32_t | Any valid float bit pattern                           | True     |
 * | upper    | Upper bound of the uniform distribution (bit-cast float)                   | uint32_t | Any valid float bit pattern                           | True     |
 */
ALWI void rrelu_tile(uint32_t idst, uint32_t lower, uint32_t upper) {
    MATH((llk_math_eltwise_unary_sfpu_rrelu<APPROX>(idst, lower, upper)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void rrelu_tile_init() { MATH((llk_math_eltwise_unary_sfpu_rrelu_init<APPROX>())); }

}  // namespace ckernel
