// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_dropout.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise dropout on each element of a of a tile in DST register at index tile_index.
 * That is each element may be zeroed out based on a given probability or it may be scaled by a given
 * scale factor. The DST register buffer must be in acquired state via *acquire_dst* call. This
 * call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index      | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     | 
 * | probability     | A non-negative integer value representing dropout probability              | uint32_t | 0 to INT_MAX (float_probability * (double) INT_MAX)   | True     | 
 * | scale_factor    | uint bitwise representation of 32 bit floating point scale factor          | uint32_t |                                                       | True     |
 */
 // clang-format on
ALWI void dropout_tile(uint32_t idst, uint32_t probability, uint32_t scale_factor) {
    MATH((llk_math_eltwise_unary_sfpu_dropout<APPROX>(idst, probability, scale_factor)));
}

/**
 * This init should be called once in kernel
 */
ALWI void dropout_kernel_init(uint32_t seed) { MATH((llk_math_eltwise_unary_sfpu_dropout_init<APPROX>(seed))); }

}  // namespace ckernel
