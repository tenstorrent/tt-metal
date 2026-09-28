// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_tanhshrink.h"
#endif

namespace ckernel {

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void tanhshrink_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_tanhshrink,
        (is_fp32_dest_acc_en, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void tanhshrink_tile_init() { MATH(SFPU_UNARY_INIT_FN(unused, sfpu::tanhshrink_init, (APPROX, is_fp32_dest_acc_en))); }

#if !defined(TT_POLY_LLK_DISABLE) && (defined(TT_POLY_TANHSHRINK_BF16_AVAILABLE))
#define TT_POLY_TANHSHRINK_BF16_ROUTE_ACTIVE 1
#else
#define TT_POLY_TANHSHRINK_BF16_ROUTE_ACTIVE 0
#endif

/** Internal BF16 typed-compiler route; public callers retain the stock entry point. */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void tanhshrink_tt_poly_bf16_tile(uint32_t idst) {
#if !TT_POLY_TANHSHRINK_BF16_ROUTE_ACTIVE
    tanhshrink_tile<is_fp32_dest_acc_en>(idst);
#else
    if constexpr (is_fp32_dest_acc_en) {
        tanhshrink_tile<is_fp32_dest_acc_en>(idst);
    } else {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            is_fp32_dest_acc_en,
            calculate_tanhshrink_tt_poly_bf16,
            (8 /* ITERATIONS */),
            idst,
            VectorMode::RC));
    }
#endif
}

/** Initialize the internal BF16 typed-compiler route. */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void tanhshrink_tt_poly_bf16_tile_init() {
#if !TT_POLY_TANHSHRINK_BF16_ROUTE_ACTIVE
    tanhshrink_tile_init<is_fp32_dest_acc_en>();
#else
    if constexpr (is_fp32_dest_acc_en) {
        tanhshrink_tile_init<is_fp32_dest_acc_en>();
    } else {
        MATH(SFPU_UNARY_INIT_FN(unused, sfpu::init_tanhshrink_tt_poly_bf16, (APPROX, is_fp32_dest_acc_en)));
    }
#endif
}

#undef TT_POLY_TANHSHRINK_BF16_ROUTE_ACTIVE

}  // namespace ckernel
