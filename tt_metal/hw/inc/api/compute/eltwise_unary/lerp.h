// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_lerp.h"
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
template <DataFormat data_format, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void lerp_tile(uint32_t idst0, uint32_t idst1, uint32_t idst2, uint32_t odst) {
    MATH((sfpu::Lerp<APPROX, data_format, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst0, idst1, idst2, odst, VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void lerp_tile_init() {
    // The init is the shared SFPU init; the data-format parameter of the op struct is irrelevant here.
    MATH((sfpu::Lerp<APPROX, DataFormat::Float16_b, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

}  // namespace ckernel
