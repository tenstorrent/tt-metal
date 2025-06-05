// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_binary_sfpu_relational_int32.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise relational operation with the two integer inputs: y = relational(x0,x1)
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
ALWI void eq_int32_tile(uint32_t idst0, uint32_t idst1) {
    MATH((llk_math_eltwise_binary_sfpu_relational_int32<APPROX, ckernel::sfpu::RelationalOp::EQ>(idst0, idst1)));
}

ALWI void ne_int32_tile(uint32_t idst0, uint32_t idst1) {
    MATH((llk_math_eltwise_binary_sfpu_relational_int32<APPROX, ckernel::sfpu::RelationalOp::NE>(idst0, idst1)));
}

ALWI void lt_int32_tile(uint32_t idst0, uint32_t idst1) {
    MATH((llk_math_eltwise_binary_sfpu_relational_int32<APPROX, ckernel::sfpu::RelationalOp::LT>(idst0, idst1)));
}

ALWI void le_int32_tile(uint32_t idst0, uint32_t idst1) {
    MATH((llk_math_eltwise_binary_sfpu_relational_int32<APPROX, ckernel::sfpu::RelationalOp::LE>(idst0, idst1)));
}

ALWI void gt_int32_tile(uint32_t idst0, uint32_t idst1) {
    MATH((llk_math_eltwise_binary_sfpu_relational_int32<APPROX, ckernel::sfpu::RelationalOp::GT>(idst0, idst1)));
}

ALWI void ge_int32_tile(uint32_t idst0, uint32_t idst1) {
    MATH((llk_math_eltwise_binary_sfpu_relational_int32<APPROX, ckernel::sfpu::RelationalOp::GE>(idst0, idst1)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void eq_int32_tile_init() {
    MATH((llk_math_eltwise_binary_sfpu_relational_int32_init<APPROX, ckernel::sfpu::RelationalOp::EQ>()));
}

ALWI void ne_int32_tile_init() {
    MATH((llk_math_eltwise_binary_sfpu_relational_int32_init<APPROX, ckernel::sfpu::RelationalOp::NE>()));
}

ALWI void lt_int32_tile_init() {
    MATH((llk_math_eltwise_binary_sfpu_relational_int32_init<APPROX, ckernel::sfpu::RelationalOp::LT>()));
}

ALWI void le_int32_tile_init() {
    MATH((llk_math_eltwise_binary_sfpu_relational_int32_init<APPROX, ckernel::sfpu::RelationalOp::LE>()));
}

ALWI void gt_int32_tile_init() {
    MATH((llk_math_eltwise_binary_sfpu_relational_int32_init<APPROX, ckernel::sfpu::RelationalOp::GT>()));
}

ALWI void ge_int32_tile_init() {
    MATH((llk_math_eltwise_binary_sfpu_relational_int32_init<APPROX, ckernel::sfpu::RelationalOp::GE>()));
}

}  // namespace ckernel
