// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_binop_with_scalar.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

// clang-format off
/**
 * Performs a simple elementwise binop with scalar operation on the input: y = binop(x,scalar)
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                                 | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform requested operation| uint32_t | Must be less than the size of the DST register buffer       | True     |
 * | mode           | 0, 1, 2, 3, and 4                                                          | uint32_t | 0, 1, 2, 3, 4 corresponding to add, mul, sub, div, and rsub | True     |
 * | param1         | fp32 value scalar encoded as uint32                                        | uint32_t | Must be less than the size of the DST register buffer       | True     |
 */
// clang-format on
enum { ADD_UNARY = 0, SUB_UNARY = 1, MUL_UNARY = 2, DIV_UNARY = 3, RSUB_UNARY = 4 };
ALWI void add_unary_tile(uint32_t idst, uint32_t param1) {
    MATH((llk_math_eltwise_unary_sfpu_binop_with_scalar<APPROX, ADD_UNARY>(idst, param1)));
}

ALWI void sub_unary_tile(uint32_t idst, uint32_t param1) {
    MATH((llk_math_eltwise_unary_sfpu_binop_with_scalar<APPROX, SUB_UNARY>(idst, param1)));
}

ALWI void mul_unary_tile(uint32_t idst, uint32_t param1) {
    MATH((llk_math_eltwise_unary_sfpu_binop_with_scalar<APPROX, MUL_UNARY>(idst, param1)));
}

ALWI void div_unary_tile(uint32_t idst, uint32_t param1) {
    MATH((llk_math_eltwise_unary_sfpu_binop_with_scalar<APPROX, DIV_UNARY>(idst, param1)));
}

ALWI void rsub_unary_tile(uint32_t idst, uint32_t param1) {
    MATH((llk_math_eltwise_unary_sfpu_binop_with_scalar<APPROX, RSUB_UNARY>(idst, param1)));
}

// clang-format off
/**
* Performs element-wise add operation with int32 scalar. The DST
* register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
* compute engine.
*
* Return value: None
*
* | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
* |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
* | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
* | param1          | int32 value scalar encoded as uint32                                       | uint32_t | Must be less than the size of the DST register buffer | True     |
*/
// clang-format on

ALWI void add_unary_tile_int32(uint32_t idst, uint32_t param1) {
    MATH((llk_math_eltwise_unary_sfpu_binop_with_scalar_add_int32<APPROX, 8>(idst, param1)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void binop_with_scalar_tile_init() { MATH((llk_math_eltwise_unary_sfpu_binop_with_scalar_init<APPROX>())); }

}  // namespace ckernel
