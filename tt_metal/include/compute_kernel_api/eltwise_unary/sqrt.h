// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */
ALWI void sqrt_tile_init() { MATH(SFPU_INIT_KERNEL_CALL(sqrt, sfpu::sqrt_init, APPROX)); }

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
template <bool FAST_APPROX = false>
ALWI void sqrt_tile(uint32_t idst) {
    MATH(SFPU_FOUR_PARAM_KERNEL_ITER_FIRST(sqrt, APPROX, 8, DST_ACCUM_MODE, FAST_APPROX, idst, RC));
}

}  // namespace ckernel
