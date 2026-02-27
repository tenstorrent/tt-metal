// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_lgamma.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs elementwise natural logarithm of the gamma function: out = lgamma(x)
 *
 * | Argument | Description                                                | Type     | Valid Range                                           | Required |
 * |----------|------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst     | Destination index                                          | uint32_t | 0 to (num_dests-1)                                    | Yes      |
 */

// clang-format on
ALWI void lgamma_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_lgamma<APPROX, DST_ACCUM_MODE>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void lgamma_tile_init() { MATH((llk_math_eltwise_unary_sfpu_lgamma_init<APPROX, DST_ACCUM_MODE>())); }

}  // namespace ckernel
