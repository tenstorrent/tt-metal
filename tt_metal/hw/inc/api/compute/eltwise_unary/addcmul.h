// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_addcmul.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs elementwise addcmul (add with constant multiply) operation.
 *
 * Mathematical formula: odst = idst0 + (value * idst1 * idst2)
 *
 * This operation computes the elementwise result by:
 *   1. Multiplying the scalar value with each element of idst1
 *   2. Multiplying that result with the corresponding element of idst2
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
ALWI void addcmul_tile(
    std::uint32_t idst0, std::uint32_t idst1, std::uint32_t idst2, std::uint32_t odst, std::uint32_t value) {
    MATH((sfpu::Addcmul<APPROX, is_fp32_dest_acc_en, data_format>::run(idst0, idst1, idst2, odst, value)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void addcmul_tile_init() { MATH((sfpu::Addcmul<APPROX>::init())); }

}  // namespace ckernel
