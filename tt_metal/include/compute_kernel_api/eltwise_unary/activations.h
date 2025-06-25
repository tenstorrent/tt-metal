// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_activations.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise hardshrink operation. The lambda is provided as const param0. The DST
 * register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
 * compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The lambda value for the Hardshrink formulation                            | float    |                                                       | True     |
 */
// clang-format on
ALWI void hardshrink_tile(uint32_t idst, float param0) {
    MATH((llk_math_eltwise_unary_sfpu_hardshrink<APPROX, ckernel::ActivationType::Hardshrink>(idst, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void hardshrink_tile_init() { MATH((llk_math_eltwise_unary_sfpu_hardshrink_init<APPROX>())); }

}  // namespace ckernel
