// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_logical_not_noti.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {
// clang-format off
/**
 * Performs element-wise computation of the logical not unary operation on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void logical_not_unary_tile(uint32_t idst) {
    MATH(SFPU_UNARY_KERNEL_THREE_TEMPLATE_ARGS(logical_not_unary, APPROX, sfpi::vFloat, float, idst));
}

ALWI void logical_not_unary_tile_int32(uint32_t idst) {
    MATH(SFPU_UNARY_KERNEL_THREE_TEMPLATE_ARGS(logical_not_unary, APPROX, sfpi::vInt, int16_t, idst));
}

ALWI void logical_not_unary_tile_uint32(uint32_t idst) {
    MATH(SFPU_UNARY_KERNEL_THREE_TEMPLATE_ARGS(logical_not_unary, APPROX, sfpi::vUInt, uint16_t, idst));
}

ALWI void logical_not_unary_tile_uint16(uint32_t idst) {
    MATH((SFPU_UNARY_NO_PARAM_KERNEL(logical_not_unary_uint16, RC, APPROX, idst)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void logical_not_unary_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(logical_not_unary, APPROX)); }

}  // namespace ckernel
