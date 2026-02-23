// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_erf_erfc.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

/************** ERF *****************/
/**
 * Please refer to documentation for any_init.
 */
template <ckernel::ApproximationMode approx_mode = APPROX_MODE>
ALWI void erf_tile_init() {
    MATH(SFPU_UNARY_KERNEL_INIT(erf, approx_mode));
}

// clang-format off
/**
 * Performs element-wise computation of error function on each element of a tile
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
template <ckernel::ApproximationMode approx_mode = APPROX_MODE>
ALWI void erf_tile(uint32_t idst) {
    MATH(SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_erf, RC, approx_mode, idst));
}

/************** ERFC *****************/

/**
 * Please refer to documentation for any_init.
 */
template <ckernel::ApproximationMode approx_mode = APPROX_MODE>
ALWI void erfc_tile_init() {
    MATH(SFPU_UNARY_KERNEL_INIT(erfc, approx_mode));
}

// clang-format off
/**
 * Performs element-wise computation of complimentary error function on each element of a tile
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
template <ckernel::ApproximationMode approx_mode = APPROX_MODE>
ALWI void erfc_tile(uint32_t idst) {
    MATH(SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_erfc, RC, approx_mode, idst));
}

}  // namespace ckernel
