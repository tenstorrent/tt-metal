// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_rsqrt.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */
template <bool legacy_compat = false>
ALWI void rsqrt_tile_init() {
    MATH(SFPU_UNARY_INIT_FN(rsqrt, sfpu::rsqrt_init, (APPROX, legacy_compat)));
}

// clang-format off
/**
 * Performs element-wise computation of reciprocal sqrt on each element of a
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
template <bool legacy_compat = false, bool FAST_APPROX = false, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void rsqrt_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_rsqrt,
        (APPROX, 8 /* ITERATIONS */, is_fp32_dest_acc_en, FAST_APPROX, legacy_compat),
        idst,
        VectorMode::RC));
}

#if !defined(TT_POLY_LLK_DISABLE) && (defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE))
#define TT_POLY_RSQRT_BF16_ROUTE_ACTIVE 1
#else
#define TT_POLY_RSQRT_BF16_ROUTE_ACTIVE 0
#endif

/** Internal BF16 typed-compiler route; public callers retain the stock entry point. */
template <bool legacy_compat = false, bool FAST_APPROX = false, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void rsqrt_tt_poly_bf16_tile(uint32_t idst) {
#if !TT_POLY_RSQRT_BF16_ROUTE_ACTIVE
    rsqrt_tile<legacy_compat, FAST_APPROX, is_fp32_dest_acc_en>(idst);
#else
    if constexpr (is_fp32_dest_acc_en) {
        rsqrt_tile<legacy_compat, FAST_APPROX, is_fp32_dest_acc_en>(idst);
    } else {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            is_fp32_dest_acc_en,
            calculate_rsqrt_tt_poly_bf16,
            (8 /* ITERATIONS */),
            idst,
            VectorMode::RC));
    }
#endif
}

/** Initialize the internal BF16 typed-compiler route. */
template <bool legacy_compat = false, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void rsqrt_tt_poly_bf16_tile_init() {
#if !TT_POLY_RSQRT_BF16_ROUTE_ACTIVE
    rsqrt_tile_init<legacy_compat>();
#else
    if constexpr (is_fp32_dest_acc_en) {
        rsqrt_tile_init<legacy_compat>();
    } else {
        MATH(SFPU_UNARY_INIT_FN(rsqrt, sfpu::init_rsqrt_tt_poly_bf16, (APPROX, legacy_compat)));
    }
#endif
}

#undef TT_POLY_RSQRT_BF16_ROUTE_ACTIVE

}  // namespace ckernel
