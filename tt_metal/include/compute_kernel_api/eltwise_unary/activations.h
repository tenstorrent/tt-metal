// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_activations.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

// clang-format off
/**
* Performs element-wise hardsigmoid operation. The DST
* register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
* compute engine.
*
* Return value: None
*
* | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
* |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
* | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
*/
// clang-format on
ALWI void hardsigmoid_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_hardsigmoid<APPROX, ckernel::ActivationType::Hardsigmoid>(idst)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void hardsigmoid_tile_init() { MATH((llk_math_eltwise_unary_sfpu_hardsigmoid_init<APPROX>())); }

// clang-format off
/**
* Performs element-wise softsign operation. The DST
* register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
* compute engine.
*
* Return value: None
*
* | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
* |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
* | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
*/
// clang-format on
ALWI void softsign_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_softsign<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void softsign_tile_init() { MATH((llk_math_eltwise_unary_sfpu_softsign_init<APPROX>())); }

// clang-format off
/**
* Performs element-wise celu operation. The DST
* register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
* compute engine.
*
* Return value: None
*
* | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
* |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
* | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
* | alpha           | The alpha parameter for the CELU function                                  | uint32_t |                                                       | True     |
* | alpha_recip     | The reciprocal of the alpha parameter for the CELU function                | uint32_t |                                                       | True     |
*/
// clang-format on
ALWI void celu_tile(uint32_t idst, uint32_t alpha, uint32_t alpha_recip) {
    MATH((llk_math_eltwise_unary_sfpu_celu<APPROX, ckernel::ActivationType::Celu>(idst, alpha, alpha_recip)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void celu_tile_init() { MATH((llk_math_eltwise_unary_sfpu_celu_init<APPROX>())); }

}  // namespace ckernel
