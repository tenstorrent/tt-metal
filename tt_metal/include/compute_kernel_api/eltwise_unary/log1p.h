// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_log1p.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */

template <bool fast_and_approx = true>
ALWI void log1p_tile_init() {
    MATH(SFPU_INIT_KERNEL_CALL_FAST_APPROX(log1p, sfpu::log1p_init, APPROX, fast_and_approx));
}

// clang-format off
/**
 * Performs element-wise computation of logarithm of (1+x) on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool fast_and_approx = true>
ALWI void log1p_tile(uint32_t idst) {
    MATH(SFPU_UNARY_NO_PARAM_KERNEL_LOG1P(log1p, RC, APPROX, fast_and_approx, idst));
}

}  // namespace ckernel
