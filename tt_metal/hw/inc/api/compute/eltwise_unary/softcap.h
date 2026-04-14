// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
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
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | cap_u           | The cap parameter as a bit-cast uint32_t                                   | uint32_t | Positive float bit pattern                            | True     |
 */
// clang-format on
ALWI void softcap_tile(uint32_t idst, uint32_t cap_u, uint32_t rcap_u) {
    MATH((llk_math_eltwise_unary_sfpu_softcap<APPROX>(idst, cap_u, rcap_u)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void softcap_tile_init() { MATH((llk_math_eltwise_unary_sfpu_softcap_init<APPROX>())); }

}  // namespace ckernel
