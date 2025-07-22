// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_binary_sfpu_mul_int.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise mul operation with the two integer inputs: y = mul(x0,x1)
 * Output overwrites first operand in DST.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
 * A maximum of 4 tiles from each operand can be loaded into DST at once, for a total of 8 tiles,
 * when using 16 bit formats. This gets reduced to 2 tiles from each operand for 32 bit formats.
 *
 * Return value: None
 *
 * | Argument              | Description                                                                 | Type     | Valid Range                                           | Required |
 * |-----------------------|-----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0                 | The index of the tile in DST register buffer to use as first operand        | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1                 | The index of the tile in DST register buffer to use as second operand       | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void mul_uint16_tile(uint32_t idst0, uint32_t idst1) {
    MATH((llk_math_eltwise_binary_sfpu_mul_int<APPROX, 8>(idst0, idst1)));
}

/**
 * Please refer to documentation for mul_uint16_tile_init.
 */
ALWI void mul_int_tile_init() { MATH((llk_math_eltwise_binary_sfpu_mul_int_init<APPROX>())); }

}  // namespace ckernel
