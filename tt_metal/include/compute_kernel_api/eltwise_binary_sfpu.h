// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_binary_sfpu_binop.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

/**
 * Performs an elementwise binop operation with the two inputs: y = binop(x0,x1)
 * Output overwrites first operand in DST.
 *
 * Return value: None
 *
 * | Argument       | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0          | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1          | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
enum { ADD_BINARY = 0, SUB_BINARY = 1, MUL_BINARY = 2, DIV_BINARY = 3, RSUB_BINARY = 4, POW_BINARY = 5 };
ALWI void add_binary_tile(uint32_t idst0, uint32_t idst1) {
    MATH((llk_math_eltwise_binary_sfpu_binop<APPROX, ADD_BINARY>(idst0, idst1)));
}

ALWI void sub_binary_tile(uint32_t idst0, uint32_t idst1) {
    MATH((llk_math_eltwise_binary_sfpu_binop<APPROX, SUB_BINARY>(idst0, idst1)));
}

ALWI void mul_binary_tile(uint32_t idst0, uint32_t idst1) {
    MATH((llk_math_eltwise_binary_sfpu_binop<APPROX, MUL_BINARY>(idst0, idst1)));
}

ALWI void div_binary_tile(uint32_t idst0, uint32_t idst1) {
    MATH((llk_math_eltwise_binary_sfpu_binop<APPROX, DIV_BINARY>(idst0, idst1)));
}

ALWI void rsub_binary_tile(uint32_t idst0, uint32_t idst1) {
    MATH((llk_math_eltwise_binary_sfpu_binop<APPROX, RSUB_BINARY>(idst0, idst1)));
}

ALWI void power_binary_tile(uint32_t idst0, uint32_t idst1) {
    MATH((llk_math_eltwise_binary_sfpu_binop<APPROX, POW_BINARY>(idst0, idst1)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void eltwise_binop_tile_init() { MATH((llk_math_eltwise_binary_sfpu_binop_init<APPROX>())); }

}  // namespace ckernel
