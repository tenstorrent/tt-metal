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
 * Performs elementwise natural logarithm of the gamma function: out = lgamma(x). lgamma is computed using Stirling approximation for x > 0.5.
 * For x < 0.5, the reflection formula (1 - x) is used. The result adjustment for (inputs < 0.5) is not part of this kernel.
 *
 * | Argument | Description                                                | Type     | Valid Range                                           | Required |
 * |----------|------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst     | Destination index                                          | uint32_t | 0 to (num_dests-1)                                    | Yes      |
 */

// clang-format on
ALWI void lgamma_stirling_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_lgamma_stirling<APPROX, DST_ACCUM_MODE>(idst)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void lgamma_stirling_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_lgamma_stirling_init<APPROX, DST_ACCUM_MODE>()));
}

/**
 * Performs elementwise natural logarithm of the gamma function: out = lgamma(x). lgamma is computed using Stirling
 * approximation for x > 0.5. For x < 0.5, the reflection formula (1 - x) is used. The result adjustment for (inputs <
 * 0.5) is not part of this kernel.
 *
 * | Argument | Description                                                | Type     | Valid Range | Required |
 * |----------|------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0    | Lgamma stirling result                                     | uint32_t | 0 to (num_dests-1) | Yes      |
 * | idst1    | log|sin(pi*x)| result with correction for integer values   | uint32_t | 0 to (num_dests-1) | Yes      |
 * | idst2    | x                                          | uint32_t      | 0 to (num_dests-1) | Yes      | | idst3 |
 * Destination index                                          | uint32_t | 0 to (num_dests-1) | Yes      |
 */

// clang-format on
ALWI void lgamma_adjusted_tile(uint32_t idst0, uint32_t idst1, uint32_t idst2, uint32_t idst3) {
    MATH((llk_math_eltwise_ternary_sfpu_lgamma_adjusted<APPROX, DST_ACCUM_MODE>(idst0, idst1, idst2, idst3)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void lgamma_adjusted_tile_init() {
    MATH((llk_math_eltwise_ternary_sfpu_lgamma_adjusted_init<APPROX, DST_ACCUM_MODE>()));
}

}  // namespace ckernel
