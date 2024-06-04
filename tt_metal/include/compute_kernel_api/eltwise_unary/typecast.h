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
 * Performs an elementwise typecast operation on the input.
 * Supports following typecasts:
 *  fp32/fp16b -> uint32
 *  fp32/fp16b -> uint16
 * For output to be uint32, Dest must be in 32 bit mode.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform typecast operation | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | OUT_DTYPE      | Desired output data format                                                 | uint32_t | Must be valid tt::DataFormat                          | True     |
 */
template <uint32_t OUT_DTYPE>
ALWI void typecast_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_typecast<APPROX, OUT_DTYPE>(idst) ));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void typecast_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_typecast_init<APPROX>() ));
}


} // namespace ckernel
