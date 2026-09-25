// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#ifndef ARCH_QUASAR
#include "ckernel_sfpu_softsign.h"
#include "ckernel_sfpu_softshrink.h"
#include "ckernel_sfpu_hardshrink.h"
#include "ckernel_sfpu_celu.h"
#include "ckernel_sfpu_activations.h"
#endif
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

#ifndef ARCH_QUASAR
// clang-format off
/**
* Performs element-wise hardsigmoid operation. The DST
* register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
* compute engine.
*
* Return value: None
*
* | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
* |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
* | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
*/
// clang-format on
ALWI void hardsigmoid_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_activation,
        (APPROX, ckernel::ActivationType::Hardsigmoid, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC));
}

ALWI void hardsigmoid_tile_pack(uint32_t idst) {
    PACK(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_activation,
        (APPROX, ckernel::ActivationType::Hardsigmoid, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void hardsigmoid_tile_init() { MATH(SFPU_UNARY_INIT_FN(hardsigmoid, sfpu::hardsigmoid_init, (APPROX))); }

ALWI void hardsigmoid_tile_init_pack() { PACK(SFPU_UNARY_INIT_FN(hardsigmoid, sfpu::hardsigmoid_init, (APPROX))); }

// clang-format off
/**
* Performs element-wise softsign operation. The DST
* register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
* compute engine.
*
* Return value: None
*
* | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
* |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
* | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
*/
// clang-format on
ALWI void softsign_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_softsign, (APPROX, 8 /* ITERATIONS */), idst, VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void softsign_tile_init() { MATH(SFPU_UNARY_INIT_FN(softsign, sfpu::init_softsign, (APPROX))); }

// clang-format off
/**
* Performs element-wise celu operation. The DST
* register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
* compute engine.
*
* Return value: None
*
* | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
* |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
* | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
* | alpha           | The alpha parameter for the CELU function                                  | uint32_t |                                                       | True     |
* | alpha_recip     | The reciprocal of the alpha parameter for the CELU function                | uint32_t |                                                       | True     |
*/
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void celu_tile(uint32_t idst, uint32_t alpha, uint32_t alpha_recip) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_celu,
        (APPROX, is_fp32_dest_acc_en, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC,
        alpha,
        alpha_recip));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void celu_tile_init() { MATH(SFPU_UNARY_INIT(celu)); }

// clang-format off
 /**
 * Performs element-wise softshrink operation. The DST
 * register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
 * compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The λ value for the Softshrink formulation                                 | uint32   |                                                       | True     |
 */
 // clang-format on
ALWI void softshrink_tile(uint32_t idst, uint32_t param0) {
     MATH(SFPU_UNARY_CALL(
         DST_SYNC_MODE,
         DST_ACCUM_MODE,
         calculate_softshrink,
         (APPROX, 8 /* ITERATIONS */),
         idst,
         VectorMode::RC,
         param0));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void softshrink_tile_init() { MATH(SFPU_UNARY_INIT(softshrink)); }

// clang-format off
/**
* Performs element-wise hardshrink operation on each element of a tile
* in DST register at index idst. The DST register buffer must be in
* acquired state via *acquire_dst* call. This call is blocking and is only
* available on the compute engine.
*
* Formula: hardshrink(x, λ) = x if |x| > λ, else 0
*
* Return value: None
*
* | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
* |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
* | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
* | param0          | The λ value for the Hardshrink formulation                                 | uint32_t |                                                       | True     |
*/
// clang-format on
ALWI void hardshrink_tile(uint32_t idst, uint32_t param0) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_hardshrink,
        (APPROX, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC,
        param0));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void hardshrink_tile_init() { MATH(SFPU_UNARY_INIT(hardshrink)); }
#endif  // !ARCH_QUASAR

#if !defined(TT_POLY_LLK_DISABLE) &&                                                                                 \
    ((defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)) && defined(TT_METAL_SFPU_SINGLE_TILE_DST) &&                \
     TT_METAL_SFPU_SINGLE_TILE_DST == 1 && defined(TT_POLY_BF16_UNARY_CONTEXT) && TT_POLY_BF16_UNARY_CONTEXT == 1 && \
     defined(SFPU_OP_PROGRAM_INIT_0))
#define TT_POLY_HARDSIGMOID_BF16_ROUTE_ACTIVE 1
#else
#define TT_POLY_HARDSIGMOID_BF16_ROUTE_ACTIVE 0
#endif

/** Internal BF16 typed-compiler route; public callers retain the stock entry point. */
ALWI void hardsigmoid_tt_poly_bf16_tile(uint32_t idst) {
#if !TT_POLY_HARDSIGMOID_BF16_ROUTE_ACTIVE
    hardsigmoid_tile(idst);
#else
    if constexpr (DST_ACCUM_MODE) {
        hardsigmoid_tile(idst);
    } else {
        if (idst != 0) {
            hardsigmoid_tile_init();
            hardsigmoid_tile(idst);
#if !TT_POLY_HARDSIGMOID_BF16_ROUTE_ACTIVE
            hardsigmoid_tile_init();
#else
            if constexpr (DST_ACCUM_MODE) {
                hardsigmoid_tile_init();
            } else {
                hardsigmoid_tile_init();
                MATH(sfpu::init_hardsigmoid_tt_poly_bf16());
            }
#endif
            return;
        }
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_hardsigmoid_tt_poly_bf16,
            (32 /* ITERATIONS */),
            idst,
            VectorMode::None));
    }
#endif
}

/** Initialize the internal BF16 typed-compiler route. */
ALWI void hardsigmoid_tt_poly_bf16_tile_init() {
#if !TT_POLY_HARDSIGMOID_BF16_ROUTE_ACTIVE
    hardsigmoid_tile_init();
#else
    if constexpr (DST_ACCUM_MODE) {
        hardsigmoid_tile_init();
    }
#endif
}

/** Initialize the selected single-tile program once, before its tile loop. */
ALWI void hardsigmoid_tt_poly_bf16_program_init() {
#if TT_POLY_HARDSIGMOID_BF16_ROUTE_ACTIVE
    if constexpr (!(DST_ACCUM_MODE)) {
#if !TT_POLY_HARDSIGMOID_BF16_ROUTE_ACTIVE
        hardsigmoid_tile_init();
#else
        if constexpr (DST_ACCUM_MODE) {
            hardsigmoid_tile_init();
        } else {
            hardsigmoid_tile_init();
            MATH(sfpu::init_hardsigmoid_tt_poly_bf16());
        }
#endif
    }
#endif
}

#undef TT_POLY_HARDSIGMOID_BF16_ROUTE_ACTIVE

#if !defined(TT_POLY_LLK_DISABLE) && (defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE))
#define TT_POLY_HARDSHRINK_BF16_ROUTE_ACTIVE 1
#else
#define TT_POLY_HARDSHRINK_BF16_ROUTE_ACTIVE 0
#endif

/** Internal BF16 typed-compiler route; public callers retain the stock entry point. */
ALWI void hardshrink_tt_poly_bf16_tile(uint32_t idst, uint32_t param0) {
#if !TT_POLY_HARDSHRINK_BF16_ROUTE_ACTIVE
    hardshrink_tile(idst, param0);
#else
    if constexpr (DST_ACCUM_MODE) {
        hardshrink_tile(idst, param0);
    } else {
        if (param0 != 0x3f000000u) {
            hardshrink_tile(idst, param0);
            return;
        }
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_hardshrink_tt_poly_bf16,
            (8 /* ITERATIONS */),
            idst,
            VectorMode::RC));
    }
#endif
}

/** Initialize the internal BF16 typed-compiler route. */
ALWI void hardshrink_tt_poly_bf16_tile_init() {
#if !TT_POLY_HARDSHRINK_BF16_ROUTE_ACTIVE
    hardshrink_tile_init();
#else
    if constexpr (DST_ACCUM_MODE) {
        hardshrink_tile_init();
    } else {
        hardshrink_tile_init();
        MATH(sfpu::init_hardshrink_tt_poly_bf16());
    }
#endif
}

#undef TT_POLY_HARDSHRINK_BF16_ROUTE_ACTIVE

}  // namespace ckernel
