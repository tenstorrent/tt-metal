// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_reshuffle_rows.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

// clang-format off
// row reshuffle
/**
 * Reshuffles the rows of the input tile to locations given by indices stored
 * at L1 address provided in idx_addr. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     | 
 * | idx_addr        | Address at which array of output row indices is stored                     | uint32_t |                                                       | True     |
 */
 // clang-format on
ALWI void reshuffle_rows_tile(uint32_t idst, uint32_t idx_addr) {
    MATH((llk_math_eltwise_unary_sfpu_reshuffle_rows<APPROX>(idst, idx_addr)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void reshuffle_rows_tile_init() { MATH((llk_math_eltwise_unary_sfpu_reshuffle_rows_init<APPROX>())); }

}  // namespace ckernel
