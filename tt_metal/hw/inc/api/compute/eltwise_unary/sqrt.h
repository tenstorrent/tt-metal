// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {
/**
 * Please refer to documentation for any_init.
 */
ALWI void sqrt_tile_init() { MATH(SFPU_UNARY_INIT_FN(sqrt, sfpu::sqrt_init, (APPROX))); }

// clang-format off
/**
 * Performs element-wise computation of the square root on each element of a
 * tile in DST register at index idst. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool FAST_APPROX = false, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sqrt_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_sqrt,
        (APPROX, 8 /*ITERATIONS*/, is_fp32_dest_acc_en, FAST_APPROX),
        idst,
        VectorMode::RC));
}

#if !defined(TT_POLY_LLK_DISABLE) &&                                                                                 \
    ((defined(TT_POLY_SQRT_BF16_AVAILABLE)) && defined(TT_METAL_SFPU_SINGLE_TILE_DST) &&                             \
     TT_METAL_SFPU_SINGLE_TILE_DST == 1 && defined(TT_POLY_BF16_UNARY_CONTEXT) && TT_POLY_BF16_UNARY_CONTEXT == 1 && \
     defined(SFPU_OP_PROGRAM_INIT_0))
#define TT_POLY_SQRT_BF16_ROUTE_ACTIVE 1
#else
#define TT_POLY_SQRT_BF16_ROUTE_ACTIVE 0
#endif

/** Internal BF16 typed-compiler route; public callers retain the stock entry point. */
template <bool FAST_APPROX = false, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sqrt_tt_poly_bf16_tile(uint32_t idst) {
#if !TT_POLY_SQRT_BF16_ROUTE_ACTIVE
    sqrt_tile<FAST_APPROX, is_fp32_dest_acc_en>(idst);
#else
    if constexpr (is_fp32_dest_acc_en) {
        sqrt_tile<FAST_APPROX, is_fp32_dest_acc_en>(idst);
    } else {
        if (idst != 0) {
            sqrt_tile_init();
            sqrt_tile<FAST_APPROX, is_fp32_dest_acc_en>(idst);
            MATH(SFPU_UNARY_INIT_FN(sqrt, sfpu::init_sqrt_tt_poly_bf16, (APPROX)));
            return;
        }
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            is_fp32_dest_acc_en,
            calculate_sqrt_tt_poly_bf16,
            (32 /* ITERATIONS */),
            idst,
            VectorMode::None));
    }
#endif
}

/** Initialize the internal BF16 typed-compiler route. */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sqrt_tt_poly_bf16_tile_init() {
#if !TT_POLY_SQRT_BF16_ROUTE_ACTIVE
    sqrt_tile_init();
#else
    if constexpr (is_fp32_dest_acc_en) {
        sqrt_tile_init();
    }
#endif
}

/** Initialize the selected single-tile program once, before its tile loop. */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sqrt_tt_poly_bf16_program_init() {
#if TT_POLY_SQRT_BF16_ROUTE_ACTIVE
    if constexpr (!(is_fp32_dest_acc_en)) {
        MATH(SFPU_UNARY_INIT_FN(sqrt, sfpu::init_sqrt_tt_poly_bf16, (APPROX)));
    }
#endif
}

#undef TT_POLY_SQRT_BF16_ROUTE_ACTIVE

}  // namespace ckernel
