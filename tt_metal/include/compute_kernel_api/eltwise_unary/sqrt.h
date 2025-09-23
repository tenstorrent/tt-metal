// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_sqrt.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */
template <bool legacy_compat = true>
ALWI void sqrt_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_sqrt_init<APPROX, legacy_compat>()));
}

// clang-format off
/**
 * Performs element-wise computation of the square root on each element of a
 * tile in DST register at index idst. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool legacy_compat = true>
ALWI void sqrt_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_sqrt<APPROX, DST_ACCUM_MODE, legacy_compat>(idst)));
}

}  // namespace ckernel
