// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_cumsum.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

// clang-format off
// Cumulative sum
/**
 * Calculates the columnwise (top to bottom) cumulative sum.
 * For multi tile comulative sum, tiles must come in NWH order (for example using reader_unary_transpose_wh) and
 * *first* must be false for all tiles where H != 0.
 * Tiles are also output in NWH order so writer_unary_transpose_wh can be used to store them correctly in L1
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     | 
 * | first           | Set true for tiles in the first row                                        | bool     |                                                       | False    |
 */
 // clang-format on
ALWI void cumsum_tile(uint32_t idst, bool first = true) {
    MATH((llk_math_eltwise_unary_sfpu_cumsum<false>(idst, first)));  // There is only non APPROXIMATE implementation
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void cumsum_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_cumsum_init<false>()));  // There is only non APPROXIMATE implementation
}

}  // namespace ckernel
