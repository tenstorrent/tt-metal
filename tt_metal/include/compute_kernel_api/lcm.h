// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_binary_sfpu_lcm.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise LCM operation on two inputs: y = lcm(x0, x1).
 * Both inputs must be int32 with values constrained to |value| ≤ 2^15-1 (32,767).
 * Output overwrites first operand in DST.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0          | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1          | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
 // clang-format on
ALWI void lcm_tile(uint32_t idst0, uint32_t idst1) {
    MATH((llk_math_eltwise_binary_sfpu_lcm<APPROX>(idst0, idst1)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void lcm_tile_init() { MATH((llk_math_eltwise_binary_sfpu_lcm_init<APPROX>())); }

}  // namespace ckernel
