// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "ckernel_sfpu_softsign.h"
#include "ckernel_sfpu_softshrink.h"
#include "ckernel_sfpu_hardshrink.h"
#include "ckernel_sfpu_celu.h"
#include "ckernel_sfpu_activations.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
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
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_activation,
        (APPROX, ckernel::ActivationType::Hardsigmoid, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC));
}

ALWI void hardsigmoid_tile_pack(uint32_t idst) {
    PACK(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_activation,
        (APPROX, ckernel::ActivationType::Hardsigmoid, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void hardsigmoid_tile_init() { MATH(SFPU_UNARY_INIT_FN(hardsigmoid, sfpu::hardsigmoid_init, (APPROX))); }

ALWI void hardsigmoid_tile_init_pack() { PACK(SFPU_UNARY_INIT_FN(hardsigmoid, sfpu::hardsigmoid_init, (APPROX))); }

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
ALWI void softsign_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_softsign, (APPROX, 8 /* ITERATIONS */), idst, VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void softsign_tile_init() { MATH(SFPU_UNARY_INIT_FN(softsign, sfpu::init_softsign, (APPROX))); }

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
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_celu,
        (APPROX, DST_ACCUM_MODE, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC,
        alpha,
        alpha_recip));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void celu_tile_init() { MATH(SFPU_UNARY_INIT(celu)); }

// clang-format off
 /**
 * Performs element-wise softshrink operation. The DST
 * register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
 * compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The λ value for the Softshrink formulation                                 | uint32   |                                                       | True     |
 */
 // clang-format on
 ALWI void softshrink_tile(uint32_t idst, uint32_t param0) {
     MATH(SFPU_UNARY_CALL(
         DST_SYNC_MODE,
         DST_ACCUM_MODE,
         calculate_softshrink,
         (APPROX, 8 /* ITERATIONS */),
         idst,
         VectorMode::RC,
         param0));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void softshrink_tile_init() { MATH(SFPU_UNARY_INIT(softshrink)); }

// clang-format off
/**
* Performs element-wise hardshrink operation on each element of a tile
* in DST register at index idst. The DST register buffer must be in
* acquired state via *acquire_dst* call. This call is blocking and is only
* available on the compute engine.
*
* Formula: hardshrink(x, λ) = x if |x| > λ, else 0
*
* Return value: None
*
* | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
* |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
* | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
* | param0          | The λ value for the Hardshrink formulation                                 | uint32_t |                                                       | True     |
*/
// clang-format on
ALWI void hardshrink_tile(uint32_t idst, uint32_t param0) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_hardshrink,
        (APPROX, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC,
        param0));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void hardshrink_tile_init() { MATH(SFPU_UNARY_INIT(hardshrink)); }

}  // namespace ckernel
