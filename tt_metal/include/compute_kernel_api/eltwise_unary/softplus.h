// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_softplus.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif



namespace ckernel {

/**
 * Performs element-wise computation of softplus (`1/beta * log(1 + exp(beta * x))`) on each element
 * of a tile in DST register at index tile_index. Any input value greater than the provided threshold
 * with return itself. The DST register buffer must be in acquired state via *acquire_dst* call. This
 * call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index      | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | beta            | Beta used in softplus calculation                                          | uint32_t | Greater than 0                                        | True     |
 * | beta_reciprocal | Reciprocal of beta (1/beta) used in softplus calculation                   | uint32_t | Greater than 0                                        | True     |
 * | threshold       | Threshold used in softplus calculation                                     | uint32_t | Greater than 0                                        | True     |
 */
ALWI void softplus_tile(uint32_t idst, uint32_t beta, uint32_t beta_reciprocal, uint32_t threshold) {
    MATH(( llk_math_eltwise_unary_sfpu_softplus<APPROX>(idst, beta, beta_reciprocal, threshold) ));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void softplus_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_softplus_init<APPROX>() ));
}

} // namespace ckernel
