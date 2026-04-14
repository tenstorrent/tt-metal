// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_softcap.h"
#endif

namespace ckernel {

/**
 * Performs element-wise softcap: cap * tanh(x / cap).
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * | Argument | Description                                              | Type     | Valid Range | Required |
 * |----------|----------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst     | Index of the tile in DST register buffer                 | uint32_t | Must be less than the size of the
 * DST register buffer | True     | | param0   | cap value (bit-cast float as uint32_t)                   | uint32_t |
 * Any positive float bit pattern                        | True     | | param1   | 1/cap value (bit-cast float as
 * uint32_t)                 | uint32_t | Any positive float bit pattern                        | True     |
 */
ALWI void softcap_tile(uint32_t idst, uint32_t param0 = 0, uint32_t param1 = 0) {
    MATH((llk_math_eltwise_unary_sfpu_softcap<APPROX>(idst, (int)VectorMode::RC, param0, param1)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void softcap_tile_init() { MATH((llk_math_eltwise_unary_sfpu_softcap_init<APPROX>())); }

}  // namespace ckernel
