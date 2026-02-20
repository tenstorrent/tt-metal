// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_ternary_sfpu_lerp.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs elementwise linear interpolation (lerp): out = input + weight * (end - input)
 *
 * | Argument | Description                                                | Type     | Valid Range                                           | Required |
 * |----------|------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0    | Index of the tile in DST register buffer (input/start)   | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1    | Index of the tile in DST register buffer (end)           | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst2    | Index of the tile in DST register buffer (weight)        | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst     | Index of the tile in DST register buffer (output)        | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <DataFormat data_format>
ALWI void lerp_tile(uint32_t idst0, uint32_t idst1, uint32_t idst2, uint32_t odst) {
    MATH((llk_math_eltwise_ternary_sfpu_lerp<APPROX, DST_ACCUM_MODE, data_format>(idst0, idst1, idst2, odst)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void lerp_tile_init() { MATH((llk_math_eltwise_ternary_sfpu_lerp_init<APPROX>())); }

}  // namespace ckernel
