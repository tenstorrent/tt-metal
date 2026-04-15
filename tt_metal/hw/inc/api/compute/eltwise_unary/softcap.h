// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_softcap.h"
#endif

namespace ckernel {

/**
 * Performs element-wise softcap: softcap(x, cap) = cap * tanh(x / cap).
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0   | The cap parameter encoded as bfloat16 in lower 16 bits of uint32          | uint32_t | Positive float                                        | True     |
 * | param1   | The 1/cap parameter encoded as bfloat16 in lower 16 bits of uint32        | uint32_t | Positive float                                        | True     |
 */
ALWI void softcap_tile(uint32_t idst, uint32_t param0, uint32_t param1) {
    MATH((llk_math_eltwise_unary_sfpu_softcap<APPROX>(idst, param0, param1)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void softcap_tile_init() { MATH((llk_math_eltwise_unary_sfpu_softcap_init<APPROX>())); }

}  // namespace ckernel
