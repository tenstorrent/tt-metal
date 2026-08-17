// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_lerp.h"
#include "llk_math_eltwise_ternary_sfpu_macros.h"
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
ALWI void lerp_tile(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t idst2, std::uint32_t odst) {
    dest_order::touch_sfpu();
    SFPU((SFPU_TERNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_lerp,
        (APPROX, DST_ACCUM_MODE, data_format, 8 /* ITERATIONS */),
        idst0,
        idst1,
        idst2,
        odst,
        VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void lerp_tile_init() { SFPU((SFPU_TERNARY_INIT(lerp))); }

}  // namespace ckernel
