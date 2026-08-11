// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_mac.h"
#include "llk_math_eltwise_ternary_sfpu_mac.h"
#include "llk_math_eltwise_ternary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs elementwise multiply-accumulate (mac): out = a * b + c
 *
 * | Argument | Description                                                | Type     | Valid Range                                           | Required |
 * |----------|------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0    | Index of the tile in DST register buffer (input a)       | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1    | Index of the tile in DST register buffer (input b)       | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst2    | Index of the tile in DST register buffer (input c)       | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst     | Index of the tile in DST register buffer (output)        | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <DataFormat data_format>
ALWI void mac_tile(uint32_t idst0, uint32_t idst1, uint32_t idst2, uint32_t odst) {
    MATH((SFPU_TERNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_mac,
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
template <DataFormat data_format>
ALWI void mac_tile_init() {
    MATH((SFPU_TERNARY_INIT_FN(mac, sfpu::mac_init, (APPROX, DST_ACCUM_MODE, data_format))));
}

}  // namespace ckernel
