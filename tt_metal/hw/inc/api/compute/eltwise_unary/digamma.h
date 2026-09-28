// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_digamma.h"
#endif

namespace ckernel {

ALWI void digamma_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_digamma, (APPROX), idst, VectorMode::RC));
}

ALWI void digamma_tile_init() { MATH(SFPU_UNARY_INIT_FN(unused, sfpu::digamma_init, (APPROX))); }

#if !defined(TT_POLY_LLK_DISABLE) &&                                                                                 \
    ((defined(TT_POLY_DIGAMMA_BF16_AVAILABLE)) && defined(TT_METAL_SFPU_SINGLE_TILE_DST) &&                          \
     TT_METAL_SFPU_SINGLE_TILE_DST == 1 && defined(TT_POLY_BF16_UNARY_CONTEXT) && TT_POLY_BF16_UNARY_CONTEXT == 1 && \
     defined(SFPU_OP_PROGRAM_INIT_0))
#define TT_POLY_DIGAMMA_BF16_ROUTE_ACTIVE 1
#else
#define TT_POLY_DIGAMMA_BF16_ROUTE_ACTIVE 0
#endif

/** Internal BF16 typed-compiler route; public callers retain the stock entry point. */
ALWI void digamma_tt_poly_bf16_tile(uint32_t idst) {
#if !TT_POLY_DIGAMMA_BF16_ROUTE_ACTIVE
    digamma_tile(idst);
#else
    if constexpr (DST_ACCUM_MODE) {
        digamma_tile(idst);
    } else {
        if (idst != 0) {
            digamma_tile_init();
            digamma_tile(idst);
            digamma_tile_init();
            MATH(sfpu::init_digamma_tt_poly_bf16());
            return;
        }
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_digamma_tt_poly_bf16,
            (32 /* ITERATIONS */),
            idst,
            VectorMode::None));
    }
#endif
}

/** Initialize the internal BF16 typed-compiler route. */
ALWI void digamma_tt_poly_bf16_tile_init() {
#if !TT_POLY_DIGAMMA_BF16_ROUTE_ACTIVE
    digamma_tile_init();
#else
    if constexpr (DST_ACCUM_MODE) {
        digamma_tile_init();
    }
#endif
}

/** Initialize the selected single-tile program once, before its tile loop. */
ALWI void digamma_tt_poly_bf16_program_init() {
#if TT_POLY_DIGAMMA_BF16_ROUTE_ACTIVE
    if constexpr (!(DST_ACCUM_MODE)) {
        digamma_tile_init();
        MATH(sfpu::init_digamma_tt_poly_bf16());
    }
#endif
}

#undef TT_POLY_DIGAMMA_BF16_ROUTE_ACTIVE

}  // namespace ckernel
