// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
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
 * Performs element-wise softcap operation: softcap(x, cap) = cap * tanh(x / cap).
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void softcap_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_softcap<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void softcap_tile_init(float cap) { MATH((llk_math_eltwise_unary_sfpu_softcap_init<APPROX>(cap))); }

}  // namespace ckernel
