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
 * Performs elementwise natural logarithm of the gamma function: out = lgamma(x). lgamma is computed using Stirling approximation.
 * For x < 0.5, the reflection formula (1 - x) is used before Stirling approximation.
 * The final reflection formula correction for (inputs < 0.5) is not part of this kernel.
 *
 *  lgamma_stirling_tile(idst); computes lgamma(x) for x >= 0.5 and computes lgamma(1-x) for x < 0.5.
 *
 * | Argument | Description                                                | Type     | Valid Range                                           | Required |
 * |----------|------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst     | Destination index                                          | uint32_t | 0 to (num_dests-1)                                    | Yes      |
 */

// clang-format on
ALWI void lgamma_stirling_tile(uint32_t idst0, uint32_t idst1, uint32_t idst2) {
    // MATH((llk_math_eltwise_unary_sfpu_lgamma_stirling<APPROX, DST_ACCUM_MODE>(idst)));
    MATH((llk_math_eltwise_binary_sfpu_lgamma_stirling<APPROX, DST_ACCUM_MODE>(idst0, idst1, idst2)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void lgamma_stirling_tile_init() {
    // MATH((llk_math_eltwise_unary_sfpu_lgamma_stirling_init<APPROX, DST_ACCUM_MODE>()));
    MATH((llk_math_eltwise_binary_sfpu_lgamma_stirling_init<APPROX, DST_ACCUM_MODE>()));
}

/**
 * Combines the Stirling-based lgamma approximation with the reflection formula correction for inputs x < 0.5.
 * Uses (1 - x) via the reflection formula and writes the adjusted lgamma result to the output tile.
 *
 * | Argument | Description                                                | Type     | Valid Range | Required |
 * |----------|------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0    | Index of the tile in DST register buffer (lgamma stirling) | uint32_t | 0 to (num_dests-1) | Yes      |
 * | idst1    | Index of the tile in DST register buffer (log|sin(pi*x)|)  | uint32_t | 0 to (num_dests-1) | Yes      |
 * | idst2    | Index of the tile in DST register buffer (input)           | uint32_t | 0 to (num_dests-1) | Yes      |
 * | idst3    | Index of the tile in DST register buffer (output)          | uint32_t | 0 to (num_dests-1) | Yes      |
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
