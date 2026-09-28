// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_prelu.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {
// clang-format off
/**
 * Performs element-wise prelu operation. The value to be prelued in the tile is provided as const param0. The DST
 * register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
 * compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | Constant value that is being multiplied if the input is lesser than 0      | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void prelu_tile(uint32_t idst, uint32_t param0) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_prelu, (APPROX), idst, VectorMode::RC, param0));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void prelu_tile_init() { MATH(SFPU_UNARY_INIT(prelu)); }

#if !defined(TT_POLY_LLK_DISABLE) &&                                                                                 \
    ((defined(TT_POLY_PRELU_BF16_AVAILABLE)) && defined(TT_METAL_SFPU_SINGLE_TILE_DST) &&                            \
     TT_METAL_SFPU_SINGLE_TILE_DST == 1 && defined(TT_POLY_BF16_UNARY_CONTEXT) && TT_POLY_BF16_UNARY_CONTEXT == 1 && \
     defined(SFPU_OP_PROGRAM_INIT_0))
#define TT_POLY_PRELU_BF16_ROUTE_ACTIVE 1
#else
#define TT_POLY_PRELU_BF16_ROUTE_ACTIVE 0
#endif

/** Internal BF16 typed-compiler route; public callers retain the stock entry point. */
ALWI void prelu_tt_poly_bf16_tile(uint32_t idst, uint32_t param0) {
#if !TT_POLY_PRELU_BF16_ROUTE_ACTIVE
    prelu_tile(idst, param0);
#else
    if constexpr (DST_ACCUM_MODE) {
        prelu_tile(idst, param0);
    } else {
        if (param0 != 0x3e800000u) {
            prelu_tile_init();
            prelu_tile(idst, param0);
            prelu_tile_init();
            MATH(sfpu::init_prelu_tt_poly_bf16());
            return;
        }
        if (idst != 0) {
            prelu_tile_init();
            prelu_tile(idst, param0);
            prelu_tile_init();
            MATH(sfpu::init_prelu_tt_poly_bf16());
            return;
        }
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_prelu_tt_poly_bf16,
            (32 /* ITERATIONS */),
            idst,
            VectorMode::None));
    }
#endif
}

/** Initialize the internal BF16 typed-compiler route. */
ALWI void prelu_tt_poly_bf16_tile_init() {
#if !TT_POLY_PRELU_BF16_ROUTE_ACTIVE
    prelu_tile_init();
#else
    if constexpr (DST_ACCUM_MODE) {
        prelu_tile_init();
    }
#endif
}

/** Initialize the selected single-tile program once, before its tile loop. */
ALWI void prelu_tt_poly_bf16_program_init() {
#if TT_POLY_PRELU_BF16_ROUTE_ACTIVE
    if constexpr (!(DST_ACCUM_MODE)) {
        prelu_tile_init();
        MATH(sfpu::init_prelu_tt_poly_bf16());
    }
#endif
}

#undef TT_POLY_PRELU_BF16_ROUTE_ACTIVE

}  // namespace ckernel
