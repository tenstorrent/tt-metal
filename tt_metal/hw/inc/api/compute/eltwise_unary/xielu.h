// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_xielu.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise xIELU computation on input x , where x is each element of a tile
 * in DST register at index tile_index. The input can be of float data type. The
 * DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on
 * the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                 | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform xIELU operation     | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | alpha_p        | alpha positive parameter                                                    | uint32_t |                                                       | True     |
 * | alpha_n        | alpha negative parameter                                                    | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void xielu_tile(uint32_t idst, uint32_t alpha_p, uint32_t alpha_n) {
    MATH(SFPU_UNARY_TWO_PARAM_KERNEL_WITH_DST_ACCUM(
        calculate_xielu, RC, APPROX, DST_ACCUM_MODE, idst, alpha_p, alpha_n));
}

ALWI void xielu_tile_init() { MATH(SFPU_INIT_KERNEL_CALL(xielu, sfpu::xielu_init, APPROX)); }

}  // namespace ckernel
