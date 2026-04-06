// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_atanh.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise inverse hyperbolic tangent: atanh(x) = 0.5 * ln((1+x)/(1-x)).
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void atanh_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_atanh<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void atanh_tile_init() { MATH((llk_math_eltwise_unary_sfpu_atanh_init<APPROX>())); }

}  // namespace ckernel
