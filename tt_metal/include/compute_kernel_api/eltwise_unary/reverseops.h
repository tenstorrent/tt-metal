// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_reverseops.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

// rpow: implemented as a composite operator
// rpow(a,k) = k**(a)

// RDIV : rdiv(x,y) = y/x
// implemented as tied multiply operator

// RSUB : rsub(x,y) = y-x

// clang-format off
/**
 * Performs element-wise computation of rsub ( rsub(x,y) = y -x) on each element of a tile and y is a constant param
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     | 
 * | param0         | Constant value that is being subtracted from                               | uint32_t |                                                       | True     |
 */
 // clang-format on
ALWI void rsub_tile(uint32_t idst, uint32_t param0) { MATH((llk_math_eltwise_unary_sfpu_rsub<APPROX>(idst, param0))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void rsub_tile_init() { MATH((llk_math_eltwise_unary_sfpu_rsub_init<APPROX>())); }

}  // namespace ckernel
