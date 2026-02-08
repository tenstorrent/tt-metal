// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_binop_with_scalar.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_rsub_int32.h"
#endif

namespace ckernel {
// RSUB : rsub(x,y) = y-x

// clang-format off
/**
 * Performs element-wise computation of rsub ( rsub(x,y) = y -x) on each element of a tile and y is a constant param
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | scalar         | Constant value that is being subtracted from                               | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void rsub_tile(uint32_t idst, uint32_t scalar) {
    MATH((llk_math_eltwise_unary_sfpu_binop_with_scalar<APPROX, RSUB_UNARY>(idst, scalar)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void rsub_tile_init() { MATH((llk_math_eltwise_unary_sfpu_binop_with_scalar_init<APPROX>())); }

// clang-format off
/**
 * Performs element-wise computation of rsub ( rsub(x,y) = y -x) on each element of a tile and y is a constant param for int32 dtype
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | scalar         | Constant value that is being subtracted from                               | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void rsub_unary_int32_tile(uint32_t idst, uint32_t scalar) {
    MATH((llk_math_eltwise_unary_sfpu_rsub_int32<APPROX>(idst, scalar)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void rsub_unary_int32_tile_init() { MATH((llk_math_eltwise_unary_sfpu_rsub_int32_init<APPROX>())); }

}  // namespace ckernel
