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
 * Uses hardware tanh for 2 ULP precision.
 *
 * The operation is decomposed into three steps:
 *   1. Multiply x by (1/cap) using SFPU
 *   2. Apply hardware tanh (LUT-based, 2 ULP)
 *   3. Multiply result by cap using SFPU
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
    // Step 1: x = x * (1/cap)
    MATH((llk_math_eltwise_unary_sfpu_softcap_pre_tanh<APPROX>(idst, inv_cap_param)));
    // Step 2: x = tanh(x) -- hardware LUT-based tanh, 2 ULP precision
    // tanh_tile is available because the compute kernel includes compute_kernel_api.h
    MATH((llk_math_eltwise_unary_sfpu_tanh<false, DST_ACCUM_MODE>(idst)));
    // Step 3: x = x * cap
    MATH((llk_math_eltwise_unary_sfpu_softcap_post_tanh<APPROX>(idst, cap_param)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void softcap_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_softcap_init<APPROX>()));
    MATH((llk_math_eltwise_unary_sfpu_tanh_init<false, DST_ACCUM_MODE>()));
}

}  // namespace ckernel
