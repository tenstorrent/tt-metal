// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_i1.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise computation of the first order modified Bessel function of the first kind on each element of a
 * tile in DST register at index tile_index. The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void i1_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_i1, (APPROX), idst, VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void i1_tile_init() { MATH(SFPU_UNARY_INIT_FN(i1, sfpu::i1_init, (APPROX))); }

#if !defined(TT_POLY_LLK_DISABLE) && ((defined(TT_POLY_I1_BF16_AVAILABLE)) && \
                                      defined(TT_METAL_SFPU_SINGLE_TILE_DST) && TT_METAL_SFPU_SINGLE_TILE_DST == 1)
#define TT_POLY_I1_BF16_ROUTE_ACTIVE 1
#else
#define TT_POLY_I1_BF16_ROUTE_ACTIVE 0
#endif

/** Internal BF16 typed-compiler route; public callers retain the stock entry point. */
ALWI void i1_tt_poly_bf16_tile(uint32_t idst) {
#if !TT_POLY_I1_BF16_ROUTE_ACTIVE
    i1_tile(idst);
#else
    if constexpr (DST_ACCUM_MODE) {
        i1_tile(idst);
    } else {
        if (idst != 0) {
            i1_tile_init();
            i1_tile(idst);
            return;
        }
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_i1_tt_poly_bf16, (32 /* ITERATIONS */), idst, VectorMode::None));
    }
#endif
}

/** Initialize the internal BF16 typed-compiler route. */
ALWI void i1_tt_poly_bf16_tile_init() { i1_tile_init(); }

#undef TT_POLY_I1_BF16_ROUTE_ACTIVE

}  // namespace ckernel
