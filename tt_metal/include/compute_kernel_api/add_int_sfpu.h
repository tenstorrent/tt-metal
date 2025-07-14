// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_binary_sfpu_add_int.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise add operation with the two integer inputs: y = add(x0,x1)
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
 * | sign_magnitude_format | Whether the Int32 values are in sign-magnitude format (not 2's complement)  | bool     |                                                       | False    |
 */
// clang-format on
ALWI void add_int32_tile(uint32_t idst0, uint32_t idst1) {
    MATH((llk_math_eltwise_binary_sfpu_add_int<APPROX, 8, InstrModLoadStore::INT32, false>(idst0, idst1)));
}

ALWI void add_uint16_tile(uint32_t idst0, uint32_t idst1) {
    MATH((llk_math_eltwise_binary_sfpu_add_int<APPROX, 8, InstrModLoadStore::LO16, false>(idst0, idst1)));
}

ALWI void add_uint32_tile(uint32_t idst0, uint32_t idst1) {
    MATH((llk_math_eltwise_binary_sfpu_add_int<APPROX, 8, InstrModLoadStore::INT32, false>(idst0, idst1)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void add_int_tile_init() { MATH((llk_math_eltwise_binary_sfpu_add_int_init<APPROX>())); }

}  // namespace ckernel
