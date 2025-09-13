// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_identity.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
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
ALWI void identity_tile(uint32_t idst) {
    MATH(SFPU_TWO_PARAM_KERNEL(calculate_identity, APPROX, 8, idst, (int)VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void identity_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(unused, APPROX)); }

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
ALWI void identity_tile_uint32(uint32_t idst) {
    MATH(SFPU_TWO_PARAM_KERNEL(calculate_identity_uint, APPROX, 8, idst, (int)VectorMode::RC));
}

}  // namespace ckernel
