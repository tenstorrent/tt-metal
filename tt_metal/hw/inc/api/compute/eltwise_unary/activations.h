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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void hardsigmoid_tile(uint32_t idst) {
    MATH((sfpu::Activation<APPROX, ckernel::ActivationType::Hardsigmoid, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC)));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void hardsigmoid_tile_pack(uint32_t idst) {
    PACK((sfpu::Activation<APPROX, ckernel::ActivationType::Hardsigmoid, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void hardsigmoid_tile_init() {
    MATH((sfpu::Activation<APPROX, ckernel::ActivationType::Hardsigmoid, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void hardsigmoid_tile_init_pack() {
    PACK((sfpu::Activation<APPROX, ckernel::ActivationType::Hardsigmoid, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void softsign_tile(uint32_t idst) {
    MATH((sfpu::Softsign<APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(idst, VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void softsign_tile_init() {
    MATH((sfpu::Softsign<APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void celu_tile(uint32_t idst, uint32_t alpha, uint32_t alpha_recip) {
    MATH((sfpu::Celu<APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(idst, VectorMode::RC, alpha, alpha_recip)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void celu_tile_init() {
    MATH((sfpu::Celu<APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void softshrink_tile(uint32_t idst, uint32_t param0) {
    MATH((sfpu::Softshrink<APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(idst, VectorMode::RC, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void softshrink_tile_init() {
    MATH((sfpu::Softshrink<APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void hardshrink_tile(uint32_t idst, uint32_t param0) {
    MATH((sfpu::Hardshrink<APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(idst, VectorMode::RC, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void hardshrink_tile_init() {
    MATH((sfpu::Hardshrink<APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

}  // namespace ckernel
