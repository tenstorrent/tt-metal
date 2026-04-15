// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_softcap.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise softcap operation: cap * tanh(x / cap).
 * Uses a piecewise polynomial approximation of tanh for precision.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | inv_cap_param   | 1/cap encoded as FP16_B in uint32_t                                        | uint32_t |                                                       | True     |
 * | cap_param       | cap encoded as FP16_B in uint32_t                                          | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void softcap_tile(uint32_t idst, uint32_t inv_cap_param, uint32_t cap_param) {
    MATH((llk_math_eltwise_unary_sfpu_softcap<APPROX>(idst, inv_cap_param, cap_param)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void softcap_tile_init() { MATH((llk_math_eltwise_unary_sfpu_softcap_init<APPROX>())); }

}  // namespace ckernel
