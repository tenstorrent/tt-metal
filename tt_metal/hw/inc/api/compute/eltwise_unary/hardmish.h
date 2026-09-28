// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_hardmish.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise computation of hardmish(x) = x * clamp(x + 2, 0, 2) / 2
 * (equivalently, x * clamp(0.5 * x + 1, 0, 1)) on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * For finite x, the piecewise form is:
 *   x <= -2  =>  0           (scale clamped to 0)
 *   x >= 0   =>  x           (scale clamped to 1)
 *   else     =>  x*(x+2)/2   (quadratic)
 *
 * Non-finite inputs follow IEEE 754 semantics; in particular, x = -inf yields NaN.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void hardmish_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, hardmish, (APPROX, 8 /* ITERATIONS */), idst, VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void hardmish_tile_init() { MATH(SFPU_UNARY_INIT(hardmish)); }

#if !defined(TT_POLY_LLK_DISABLE) && (defined(TT_POLY_HARDMISH_BF16_AVAILABLE))
#define TT_POLY_HARDMISH_BF16_ROUTE_ACTIVE 1
#else
#define TT_POLY_HARDMISH_BF16_ROUTE_ACTIVE 0
#endif

/** Internal BF16 typed-compiler route; public callers retain the stock entry point. */
ALWI void hardmish_tt_poly_bf16_tile(uint32_t idst) {
#if !TT_POLY_HARDMISH_BF16_ROUTE_ACTIVE
    hardmish_tile(idst);
#else
    if constexpr (DST_ACCUM_MODE) {
        hardmish_tile(idst);
    } else {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_hardmish_tt_poly_bf16,
            (8 /* ITERATIONS */),
            idst,
            VectorMode::RC));
    }
#endif
}

/** Initialize the internal BF16 typed-compiler route. */
ALWI void hardmish_tt_poly_bf16_tile_init() {
#if !TT_POLY_HARDMISH_BF16_ROUTE_ACTIVE
    hardmish_tile_init();
#else
    if constexpr (DST_ACCUM_MODE) {
        hardmish_tile_init();
    } else {
        hardmish_tile_init();
        MATH(sfpu::init_hardmish_tt_poly_bf16());
    }
#endif
}

#undef TT_POLY_HARDMISH_BF16_ROUTE_ACTIVE

}  // namespace ckernel
