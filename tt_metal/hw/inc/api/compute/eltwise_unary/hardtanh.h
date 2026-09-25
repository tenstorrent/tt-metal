// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#ifndef ARCH_QUASAR
#include "ckernel_sfpu_hardtanh.h"
#endif
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

#ifndef ARCH_QUASAR
// clang-format off
 /**
 * Performs element-wise hardtanh operation. The DST
 * register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
 * compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The minimum value of the linear region range                               | uint32_t |                                                       | True     |
 * | param1          | The maximum value of the linear region range                               | uint32_t |                                                       | True     |

 */
// clang-format on
ALWI void hardtanh_tile(uint32_t idst, uint32_t param0, uint32_t param1) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_hardtanh,
        (APPROX, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC,
        param0,
        param1));
}

ALWI void hardtanh_tile_pack(uint32_t idst, uint32_t param0, uint32_t param1) {
    PACK(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_hardtanh,
        (APPROX, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC,
        param0,
        param1));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void hardtanh_tile_init() { MATH(SFPU_UNARY_INIT(hardtanh)); }

ALWI void hardtanh_tile_init_pack() { PACK(SFPU_UNARY_INIT(hardtanh)); }
#endif  // !ARCH_QUASAR

#if !defined(TT_POLY_LLK_DISABLE) &&                                                                                 \
    ((defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)) && defined(TT_METAL_SFPU_SINGLE_TILE_DST) &&                \
     TT_METAL_SFPU_SINGLE_TILE_DST == 1 && defined(TT_POLY_BF16_UNARY_CONTEXT) && TT_POLY_BF16_UNARY_CONTEXT == 1 && \
     defined(SFPU_OP_PROGRAM_INIT_0))
#define TT_POLY_HARDTANH_BF16_ROUTE_ACTIVE 1
#else
#define TT_POLY_HARDTANH_BF16_ROUTE_ACTIVE 0
#endif

/** Internal BF16 typed-compiler route; public callers retain the stock entry point. */
ALWI void hardtanh_tt_poly_bf16_tile(uint32_t idst, uint32_t param0, uint32_t param1) {
#if !TT_POLY_HARDTANH_BF16_ROUTE_ACTIVE
    hardtanh_tile(idst, param0, param1);
#else
    if constexpr (DST_ACCUM_MODE) {
        hardtanh_tile(idst, param0, param1);
    } else {
        if (param0 != 0xbf800000u || param1 != 0x3f800000u) {
            hardtanh_tile_init();
            hardtanh_tile(idst, param0, param1);
#if !TT_POLY_HARDTANH_BF16_ROUTE_ACTIVE
            hardtanh_tile_init();
#else
            if constexpr (DST_ACCUM_MODE) {
                hardtanh_tile_init();
            } else {
                hardtanh_tile_init();
                MATH(sfpu::init_hardtanh_tt_poly_bf16());
            }
#endif
            return;
        }
        if (idst != 0) {
            hardtanh_tile_init();
            hardtanh_tile(idst, param0, param1);
#if !TT_POLY_HARDTANH_BF16_ROUTE_ACTIVE
            hardtanh_tile_init();
#else
            if constexpr (DST_ACCUM_MODE) {
                hardtanh_tile_init();
            } else {
                hardtanh_tile_init();
                MATH(sfpu::init_hardtanh_tt_poly_bf16());
            }
#endif
            return;
        }
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_hardtanh_tt_poly_bf16,
            (32 /* ITERATIONS */),
            idst,
            VectorMode::None));
    }
#endif
}

/** Initialize the internal BF16 typed-compiler route. */
ALWI void hardtanh_tt_poly_bf16_tile_init() {
#if !TT_POLY_HARDTANH_BF16_ROUTE_ACTIVE
    hardtanh_tile_init();
#else
    if constexpr (DST_ACCUM_MODE) {
        hardtanh_tile_init();
    }
#endif
}

/** Initialize the selected single-tile program once, before its tile loop. */
ALWI void hardtanh_tt_poly_bf16_program_init() {
#if TT_POLY_HARDTANH_BF16_ROUTE_ACTIVE
    if constexpr (!(DST_ACCUM_MODE)) {
#if !TT_POLY_HARDTANH_BF16_ROUTE_ACTIVE
        hardtanh_tile_init();
#else
        if constexpr (DST_ACCUM_MODE) {
            hardtanh_tile_init();
        } else {
            hardtanh_tile_init();
            MATH(sfpu::init_hardtanh_tt_poly_bf16());
        }
#endif
    }
#endif
}

#undef TT_POLY_HARDTANH_BF16_ROUTE_ACTIVE

}  // namespace ckernel
