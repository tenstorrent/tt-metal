// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_erf_erfc.h"
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
ALWI void erf_tile_init(bool fast_and_approx = true) {
    if (fast_and_approx) {
        MATH((llk_math_eltwise_unary_sfpu_erf_init<true>()));
    } else {
        MATH((llk_math_eltwise_unary_sfpu_erf_init<false>()));
    }
}


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
ALWI void erf_tile(uint32_t idst, bool fast_and_approx = true) {
    if (fast_and_approx) {
        MATH(( llk_math_eltwise_unary_sfpu_erf<true>(idst) ));
    } else {
        MATH(( llk_math_eltwise_unary_sfpu_erf<false>(idst) ));
    }
}

/************** ERFC *****************/

/**
 * Please refer to documentation for any_init.
 */
ALWI void erfc_tile_init(bool fast_and_approx = true) {
    if (fast_and_approx) {
        MATH((llk_math_eltwise_unary_sfpu_erfc_init<true>()));
    } else {
        MATH((llk_math_eltwise_unary_sfpu_erfc_init<false>()));
    }
}


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
ALWI void erfc_tile(uint32_t idst, bool fast_and_approx = true) {
    if (fast_and_approx) {
        MATH(( llk_math_eltwise_unary_sfpu_erfc<true>(idst) ));
    } else {
        MATH(( llk_math_eltwise_unary_sfpu_erfc<false>(idst) ));
    }
}

}  // namespace ckernel
