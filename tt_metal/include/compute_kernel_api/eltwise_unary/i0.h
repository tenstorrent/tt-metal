// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_i0.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {
// clang-format off
/**
 * Performs element-wise computation of the zeroth order modified Bessel function of the first kind on each element of a
 * tile in DST register at index tile_index. The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void i0_tile(uint32_t idst) { MATH(SFPU_UNARY_NO_PARAM_KERNEL(i0, RC, APPROX, idst)); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void i0_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(i0, APPROX)); }
}  // namespace ckernel
