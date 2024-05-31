// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_typecast.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif



namespace ckernel {

/**
 * Performs an elementwise typecast operation on the input
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform typecast operation | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
ALWI void typecast_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_typecast<APPROX>(idst) ));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void typecast_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_typecast_init<APPROX>() ));
}


} // namespace ckernel
