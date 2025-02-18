// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_identity.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

// clang-format off
/**
 * Performs a simple elementwise copy / identity operation on the input: y(x) = x
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform identity operation | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
 // clang-format on
ALWI void identity_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_identity<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void identity_tile_init() { MATH((llk_math_eltwise_unary_sfpu_identity_init<APPROX>())); }

// clang-format off
/**
 * Performs a simple elementwise copy / identity operation on the input: y(x) = x
 * This function should be used with unsigned integer formats: uint32
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform identity operation | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
 // clang-format on
ALWI void identity_tile_uint32(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_identity_uint32<APPROX>(idst))); }

}  // namespace ckernel
