// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_frac.h"
#endif

namespace ckernel {

// clang-format off
 /**
 * Performs element-wise frac operation: x - trunc(x).
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void frac_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_frac<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void frac_tile_init() { MATH((llk_math_eltwise_unary_sfpu_frac_init<APPROX>())); }

}  // namespace ckernel
