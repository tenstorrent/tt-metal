// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_ternary_sfpu_addcmul.h"
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
template <DataFormat data_format>
ALWI void addcmul_tile(uint32_t idst0, uint32_t idst1, uint32_t idst2, uint32_t odst, uint32_t value) {
    MATH(
        (llk_math_eltwise_ternary_sfpu_addcmul<APPROX, DST_ACCUM_MODE, data_format>(idst0, idst1, idst2, odst, value)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void addcmul_tile_init() { MATH((llk_math_eltwise_ternary_sfpu_addcmul_init<APPROX>())); }

}  // namespace ckernel
