// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#ifndef ARCH_QUASAR
#include "ckernel_sfpu_softsign.h"
#include "ckernel_sfpu_softshrink.h"
#include "ckernel_sfpu_hardshrink.h"
#include "ckernel_sfpu_celu.h"
#include "ckernel_sfpu_activations.h"
#endif
#endif

namespace ckernel {

// Quasar has none of these kernels.
#ifndef ARCH_QUASAR
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
ALWI void hardsigmoid_tile(std::uint32_t idst) { MATH((sfpu::Hardsigmoid<APPROX>::run(idst))); }

ALWI void hardsigmoid_tile_pack(std::uint32_t idst) { PACK((sfpu::Hardsigmoid<APPROX>::run(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void hardsigmoid_tile_init() { MATH((sfpu::Hardsigmoid<APPROX>::init())); }

ALWI void hardsigmoid_tile_init_pack() { PACK((sfpu::Hardsigmoid<APPROX>::init())); }

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
ALWI void softsign_tile(std::uint32_t idst) { MATH((sfpu::Softsign<APPROX>::run(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void softsign_tile_init() { MATH((sfpu::Softsign<APPROX>::init())); }

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
ALWI void celu_tile(std::uint32_t idst, std::uint32_t alpha, std::uint32_t alpha_recip) {
    MATH((sfpu::Celu<APPROX, is_fp32_dest_acc_en>::run(idst, alpha, alpha_recip)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void celu_tile_init() { MATH((sfpu::Celu<APPROX, DST_ACCUM_MODE>::init())); }

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
ALWI void softshrink_tile(std::uint32_t idst, std::uint32_t param0) {
    MATH((sfpu::Softshrink<APPROX>::run(idst, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void softshrink_tile_init() { MATH((sfpu::Softshrink<APPROX>::init())); }

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
ALWI void hardshrink_tile(std::uint32_t idst, std::uint32_t param0) {
    MATH((sfpu::Hardshrink<APPROX>::run(idst, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void hardshrink_tile_init() { MATH((sfpu::Hardshrink<APPROX>::init())); }
#endif  // !ARCH_QUASAR

}  // namespace ckernel
