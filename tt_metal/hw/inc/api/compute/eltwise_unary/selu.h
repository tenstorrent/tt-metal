// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#ifndef ARCH_QUASAR
#include "ckernel_sfpu_selu.h"
#endif
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

#ifndef ARCH_QUASAR
// clang-format off
/**
 * Performs element-wise computation of selu = scale * (max(0,x) + min(0, alpha * (exp(x)-1))), where x is each
 * element of a tile in DST register at index tile_index. scale and alpha are each passed as the raw bits of a
 * float. The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is
 * only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | scale           | Scale used in selu calculation, as the raw bits of a float                 | uint32_t |                                                       | True     |
 * | alpha           | Alpha used in selu calculation, as the raw bits of a float                 | uint32_t |                                                       | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void selu_tile(uint32_t idst, uint32_t scale, uint32_t alpha) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_selu,
        (APPROX, is_fp32_dest_acc_en, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC,
        scale,
        alpha));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void selu_tile_pack(uint32_t idst, uint32_t scale, uint32_t alpha) {
    PACK(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_selu,
        (APPROX, is_fp32_dest_acc_en, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC,
        scale,
        alpha));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void selu_tile_init() { MATH(SFPU_UNARY_INIT(selu)); }

ALWI void selu_tile_init_pack() { PACK(SFPU_UNARY_INIT(selu)); }
#endif  // !ARCH_QUASAR

#if !defined(TT_POLY_LLK_DISABLE) && ((defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)) && \
                                      defined(TT_METAL_SFPU_SINGLE_TILE_DST) && TT_METAL_SFPU_SINGLE_TILE_DST == 1)
#define TT_POLY_SELU_BF16_ROUTE_ACTIVE 1
#else
#define TT_POLY_SELU_BF16_ROUTE_ACTIVE 0
#endif

/** Internal BF16 typed-compiler route; public callers retain the stock entry point. */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void selu_tt_poly_bf16_tile(uint32_t idst, uint32_t param0, uint32_t param1) {
#if !TT_POLY_SELU_BF16_ROUTE_ACTIVE
    selu_tile<is_fp32_dest_acc_en>(idst, param0, param1);
#else
    if constexpr (is_fp32_dest_acc_en) {
        selu_tile<is_fp32_dest_acc_en>(idst, param0, param1);
    } else {
        if (param0 != 0x3f867d5fu || param1 != 0x3fd62d7du) {
            selu_tile_init();
            selu_tile<is_fp32_dest_acc_en>(idst, param0, param1);
            return;
        }
        if (idst != 0) {
            selu_tile_init();
            selu_tile<is_fp32_dest_acc_en>(idst, param0, param1);
            return;
        }
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            is_fp32_dest_acc_en,
            calculate_selu_tt_poly_bf16,
            (32 /* ITERATIONS */),
            idst,
            VectorMode::None));
    }
#endif
}

/** Initialize the internal BF16 typed-compiler route. */
ALWI void selu_tt_poly_bf16_tile_init() { selu_tile_init(); }

#undef TT_POLY_SELU_BF16_ROUTE_ACTIVE

}  // namespace ckernel
