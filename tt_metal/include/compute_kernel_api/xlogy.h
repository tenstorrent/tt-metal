// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_binary_sfpu_xlogy.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise xlogy operation y = xlogy(x0, x1) with x0 as first operand and x1 as second operand.
 * Output overwrites first operand in DST.
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
 * compute engine.
 *
 * Return value: None
 *
 * | Argument              | Description                                                                 | Type     | Valid Range                                           | Required |
 * |-----------------------|-----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0                 | The index of the tile in DST register buffer to use as first operand        | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1                 | The index of the tile in DST register buffer to use as second operand       | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void xlogy_binary_tile(uint32_t idst0, uint32_t idst1) {
    MATH((llk_math_eltwise_binary_sfpu_xlogy<APPROX>(idst0, idst1)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void xlogy_binary_tile_init() { MATH((llk_math_eltwise_binary_sfpu_xlogy_init<APPROX>())); }

}  // namespace ckernel
