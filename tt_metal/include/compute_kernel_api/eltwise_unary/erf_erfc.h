// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_erf_erfc.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

/************** ERF *****************/
/**
 * Please refer to documentation for any_init.
 */
template <bool fast_and_approx = true>
ALWI void erf_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::erf, fast_and_approx>()));
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
template <bool fast_and_approx = true>
ALWI void erf_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_params<fast_and_approx>(
        ckernel::sfpu::calculate_sfpu_erf_erfc<SfpuType::erf, fast_and_approx>, idst, (int)VectorMode::RC)));
}

/************** ERFC *****************/

/**
 * Please refer to documentation for any_init.
 */
template <bool fast_and_approx = true>
ALWI void erfc_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::erfc, fast_and_approx>()));
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
template <bool fast_and_approx = true>
ALWI void erfc_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_params<fast_and_approx>(
        ckernel::sfpu::calculate_sfpu_erf_erfc<SfpuType::erfc, fast_and_approx>, idst, (int)VectorMode::RC)));
}

}  // namespace ckernel
