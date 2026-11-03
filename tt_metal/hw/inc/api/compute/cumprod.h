// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_cumprod.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Calculates the columnwise (top to bottom) cumulative product. Same tile order, *first* flag and restrictions as
 * *cumsum_tile*. Only available on Wormhole and Blackhole. Not supported on Quasar (compile error).
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | first           | Set true for tiles in the first row                                        | bool     |                                                       | False    |
 */
// clang-format on
ALWI void cumprod_tile(uint32_t idst, bool first = true) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_cumprod,
        (false /* APPROXIMATE */),
        idst,
        VectorMode::RC_custom,
        first));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void cumprod_tile_init() { MATH(SFPU_UNARY_INIT_FN(cumprod, sfpu::cumprod_init, (false /* APPROXIMATE */))); }

}  // namespace ckernel
