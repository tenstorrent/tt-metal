// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#include "llk_math_eltwise_unary_sfpu_digamma.h"

namespace ckernel {

// clang-format off
/**
 * Performs elementwise digamma (logarithmic derivative of the gamma function) on each
 * element of a tile in DST register at index tile_index. Uses a Stirling asymptotic
 * series with upward recurrence: double recurrence for z < 2 (ψ(z) = ψ(z+2) − 1/z − 1/(z+1))
 * and single recurrence for z < 3 (ψ(z) = ψ(z+1) − 1/z) for improved accuracy.
 *
 * Valid for inputs > 0. The DST register buffer must be in acquired state via *acquire_dst*
 * call. This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument   | Description                                                                | Type     | Valid Range                                           | Required |
 * |------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void digamma_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_digamma<APPROX, DST_ACCUM_MODE>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void digamma_tile_init() { MATH((llk_math_eltwise_unary_sfpu_digamma_init<APPROX, DST_ACCUM_MODE>())); }

}  // namespace ckernel
