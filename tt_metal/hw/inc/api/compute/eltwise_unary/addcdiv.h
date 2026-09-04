// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_addcdiv.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs elementwise addcdiv operation.
 *
 * Mathematical formula: odst = idst0 + (value * idst1 / idst2)
 *
 * This operation computes the elementwise result by:
 *   1. Multiplying the scalar value with each element of idst1
 *   2. Dividing that result by the corresponding element of idst2
 *   3. Adding the result to the corresponding element of idst0
 *
 * Output overwrites odst in DST.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * | Argument | Description                                                   | Type     | Valid Range                                           | Required |
 * |----------|---------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0    | Index of the tile in DST register buffer (first input)        | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1    | Index of the tile in DST register buffer (second input)       | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst2    | Index of the tile in DST register buffer (third input)        | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | value    | Scalar constant multiplier                                    | uint32_t | Any valid value                                       | True     |
 * | odst     | Index of the tile in DST register buffer (output)             | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <DataFormat data_format, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void addcdiv_tile(uint32_t idst0, uint32_t idst1, uint32_t idst2, uint32_t odst, uint32_t value) {
    MATH((sfpu::Addcdiv<APPROX, data_format, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst0, idst1, idst2, odst, VectorMode::RC, value)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void addcdiv_tile_init() {
    // init_addcdiv only depends on APPROX; the data-format parameter of the op struct is irrelevant here.
    MATH((sfpu::Addcdiv<APPROX, DataFormat::Float16_b, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

}  // namespace ckernel
