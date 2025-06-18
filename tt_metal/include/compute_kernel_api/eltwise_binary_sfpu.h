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

// clang-format off
/**
 * Performs an elementwise binop operation with the two floating point inputs: y = binop(x0,x1)
 * Output overwrites first operand in DST.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
 * A maximum of 4 tiles from each operand can be loaded into DST at once, for a total of 8 tiles,
 * when using 16 bit formats. This gets reduced to 2 tiles from each operand for 32 bit formats.
 *
 * Return value: None
 *
 * | Argument       | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0          | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1          | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void add_binary_tile(uint32_t idst0, uint32_t idst1) {
    MATH((llk_math_eltwise_binary_sfpu_binop<APPROX, ckernel::BinaryOp::ADD>(idst0, idst1)));
}

ALWI void sub_binary_tile(uint32_t idst0, uint32_t idst1) {
    MATH((llk_math_eltwise_binary_sfpu_binop<APPROX, ckernel::BinaryOp::SUB>(idst0, idst1)));
}

ALWI void mul_binary_tile(uint32_t idst0, uint32_t idst1) {
    MATH((llk_math_eltwise_binary_sfpu_binop<APPROX, ckernel::BinaryOp::MUL>(idst0, idst1)));
}

ALWI void div_binary_tile(uint32_t idst0, uint32_t idst1) {
    MATH((llk_math_eltwise_binary_sfpu_binop<APPROX, ckernel::BinaryOp::DIV>(idst0, idst1)));
}

ALWI void rsub_binary_tile(uint32_t idst0, uint32_t idst1) {
    MATH((llk_math_eltwise_binary_sfpu_binop<APPROX, ckernel::BinaryOp::RSUB>(idst0, idst1)));
}

ALWI void power_binary_tile(uint32_t idst0, uint32_t idst1) {
    MATH((llk_math_eltwise_binary_sfpu_binop<APPROX, ckernel::BinaryOp::POW>(idst0, idst1)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void add_binary_tile_init() { MATH((llk_math_eltwise_binary_sfpu_binop_init<APPROX, ckernel::BinaryOp::ADD>())); }

ALWI void sub_binary_tile_init() { MATH((llk_math_eltwise_binary_sfpu_binop_init<APPROX, ckernel::BinaryOp::SUB>())); }

ALWI void mul_binary_tile_init() { MATH((llk_math_eltwise_binary_sfpu_binop_init<APPROX, ckernel::BinaryOp::MUL>())); }

ALWI void div_binary_tile_init() { MATH((llk_math_eltwise_binary_sfpu_binop_init<APPROX, ckernel::BinaryOp::DIV>())); }

ALWI void rsub_binary_tile_init() {
    MATH((llk_math_eltwise_binary_sfpu_binop_init<APPROX, ckernel::BinaryOp::RSUB>()));
}

ALWI void power_binary_tile_init() {
    MATH((llk_math_eltwise_binary_sfpu_binop_init<APPROX, ckernel::BinaryOp::POW>()));
}

}  // namespace ckernel
