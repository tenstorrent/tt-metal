// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_gelu.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */
template <ckernel::ApproximationMode approx_mode = ckernel::ApproximationMode::FastApproximate>
ALWI void gelu_tile_init() {
    MATH(SFPU_INIT_KERNEL_CALL(gelu, sfpu::gelu_init, approx_mode));
}

// clang-format off
/**
 * Performs element-wise computation of gelu  on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument         | Description                                                                | Type     | Valid Range                                           | Required |
 * |------------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index       | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | approx_mode      | Approximation mode selection                                               | ApproximationMode | Precise, Approximate, FastApproximate, FastApproximateClamped | False |
 */
// clang-format on
template <ckernel::ApproximationMode approx_mode = ckernel::ApproximationMode::FastApproximate>
ALWI void gelu_tile(uint32_t idst) {
    MATH(SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_gelu, RC, approx_mode, idst));
}

// TODO: Add gelu_derivative

}  // namespace ckernel
