// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#ifndef ARCH_QUASAR
#include "ckernel_sfpu_erf.h"
#include "ckernel_sfpu_erfc.h"
#endif
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {
#ifndef ARCH_QUASAR
/************** ERF *****************/
/**
 * Please refer to documentation for any_init.
 */
template <bool fast_and_approx = true>
ALWI void erf_tile_init() {
    MATH(SFPU_UNARY_INIT_FN(erf, sfpu::erf_init, (fast_and_approx)));
}

// clang-format off
/**
 * Performs element-wise computation of error function on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool fast_and_approx = true>
ALWI void erf_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_erf, (fast_and_approx), idst, VectorMode::RC));
}

/************** ERFC *****************/

/**
 * Please refer to documentation for any_init.
 */
ALWI void erfc_tile_init() { MATH(SFPU_UNARY_INIT_FN(erfc, sfpu::erfc_init, (true /*APPROXIMATION_MODE*/))); }

// clang-format off
/**
 * Performs element-wise computation of complimentary error function on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void erfc_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_erfc, (), idst, VectorMode::RC));
}

#endif

#if !defined(TT_POLY_LLK_DISABLE) &&                                                                                 \
    ((defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)) && defined(TT_METAL_SFPU_SINGLE_TILE_DST) &&                \
     TT_METAL_SFPU_SINGLE_TILE_DST == 1 && defined(TT_POLY_BF16_UNARY_CONTEXT) && TT_POLY_BF16_UNARY_CONTEXT == 1 && \
     defined(SFPU_OP_PROGRAM_INIT_0))
#define TT_POLY_ERF_BF16_ROUTE_ACTIVE 1
#else
#define TT_POLY_ERF_BF16_ROUTE_ACTIVE 0
#endif

/** Internal BF16 typed-compiler route; public callers retain the stock entry point. */
template <bool fast_and_approx = true>
ALWI void erf_tt_poly_bf16_tile(uint32_t idst) {
#if !TT_POLY_ERF_BF16_ROUTE_ACTIVE
    erf_tile<fast_and_approx>(idst);
#else
    if constexpr (DST_ACCUM_MODE) {
        erf_tile<fast_and_approx>(idst);
    } else {
        if (idst != 0) {
            erf_tile_init<fast_and_approx>();
            erf_tile<fast_and_approx>(idst);
#if !TT_POLY_ERF_BF16_ROUTE_ACTIVE
            erf_tile_init<fast_and_approx>();
#else
            if constexpr (DST_ACCUM_MODE) {
                erf_tile_init<fast_and_approx>();
            } else {
                MATH(SFPU_UNARY_INIT_FN(erf, sfpu::init_erf_tt_poly_bf16, (fast_and_approx)));
            }
#endif
            return;
        }
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_erf_tt_poly_bf16, (32 /* ITERATIONS */), idst, VectorMode::None));
    }
#endif
}

/** Initialize the internal BF16 typed-compiler route. */
template <bool fast_and_approx = true>
ALWI void erf_tt_poly_bf16_tile_init() {
#if !TT_POLY_ERF_BF16_ROUTE_ACTIVE
    erf_tile_init<fast_and_approx>();
#else
    if constexpr (DST_ACCUM_MODE) {
        erf_tile_init<fast_and_approx>();
    }
#endif
}

/** Initialize the selected single-tile program once, before its tile loop. */
template <bool fast_and_approx = true>
ALWI void erf_tt_poly_bf16_program_init() {
#if TT_POLY_ERF_BF16_ROUTE_ACTIVE
    if constexpr (!(DST_ACCUM_MODE)) {
#if !TT_POLY_ERF_BF16_ROUTE_ACTIVE
        erf_tile_init<fast_and_approx>();
#else
        if constexpr (DST_ACCUM_MODE) {
            erf_tile_init<fast_and_approx>();
        } else {
            MATH(SFPU_UNARY_INIT_FN(erf, sfpu::init_erf_tt_poly_bf16, (fast_and_approx)));
        }
#endif
    }
#endif
}

#undef TT_POLY_ERF_BF16_ROUTE_ACTIVE

#if !defined(TT_POLY_LLK_DISABLE) &&                                                                                 \
    ((defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)) && defined(TT_METAL_SFPU_SINGLE_TILE_DST) &&                \
     TT_METAL_SFPU_SINGLE_TILE_DST == 1 && defined(TT_POLY_BF16_UNARY_CONTEXT) && TT_POLY_BF16_UNARY_CONTEXT == 1 && \
     defined(SFPU_OP_PROGRAM_INIT_0))
#define TT_POLY_ERFC_BF16_ROUTE_ACTIVE 1
#else
#define TT_POLY_ERFC_BF16_ROUTE_ACTIVE 0
#endif

/** Internal BF16 typed-compiler route; public callers retain the stock entry point. */
ALWI void erfc_tt_poly_bf16_tile(uint32_t idst) {
#if !TT_POLY_ERFC_BF16_ROUTE_ACTIVE
    erfc_tile(idst);
#else
    if constexpr (DST_ACCUM_MODE) {
        erfc_tile(idst);
    } else {
        if (idst != 0) {
            erfc_tile_init();
            erfc_tile(idst);
#if !TT_POLY_ERFC_BF16_ROUTE_ACTIVE
            erfc_tile_init();
#else
            if constexpr (DST_ACCUM_MODE) {
                erfc_tile_init();
            } else {
                MATH(SFPU_UNARY_INIT_FN(erfc, sfpu::init_erfc_tt_poly_bf16, (true /*APPROXIMATION_MODE*/)));
            }
#endif
            return;
        }
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_erfc_tt_poly_bf16, (32 /* ITERATIONS */), idst, VectorMode::None));
    }
#endif
}

/** Initialize the internal BF16 typed-compiler route. */
ALWI void erfc_tt_poly_bf16_tile_init() {
#if !TT_POLY_ERFC_BF16_ROUTE_ACTIVE
    erfc_tile_init();
#else
    if constexpr (DST_ACCUM_MODE) {
        erfc_tile_init();
    }
#endif
}

/** Initialize the selected single-tile program once, before its tile loop. */
ALWI void erfc_tt_poly_bf16_program_init() {
#if TT_POLY_ERFC_BF16_ROUTE_ACTIVE
    if constexpr (!(DST_ACCUM_MODE)) {
#if !TT_POLY_ERFC_BF16_ROUTE_ACTIVE
        erfc_tile_init();
#else
        if constexpr (DST_ACCUM_MODE) {
            erfc_tile_init();
        } else {
            MATH(SFPU_UNARY_INIT_FN(erfc, sfpu::init_erfc_tt_poly_bf16, (true /*APPROXIMATION_MODE*/)));
        }
#endif
    }
#endif
}

#undef TT_POLY_ERFC_BF16_ROUTE_ACTIVE

}  // namespace ckernel
