// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_hardswish.h"
#endif

namespace ckernel {

// clang-format off
 /**
 * Performs element-wise hardswish operation: x * min(max(x + 3, 0), 6) / 6.
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
ALWI void hardswish_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_hardswish<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void hardswish_tile_init() { MATH((llk_math_eltwise_unary_sfpu_hardswish_init<APPROX>())); }

}  // namespace ckernel
