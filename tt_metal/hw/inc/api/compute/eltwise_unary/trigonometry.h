// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_trigonometry.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */
ALWI void sin_tile_init() { MATH(SFPU_UNARY_INIT_FN(sine, ckernel::sfpu::sine_init, (APPROX))); }

// clang-format off
/**
 * Performs element-wise computation of the trigonometric sine operation on each element of a tile
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sin_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_sine,
        (APPROX, is_fp32_dest_acc_en, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void cos_tile_init() { MATH(SFPU_UNARY_INIT_FN(cosine, ckernel::sfpu::cosine_init, (APPROX))); }

// clang-format off
/**
 * Performs element-wise computation of the trigonometric cosine operation on each element of a tile
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void cos_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_cosine,
        (APPROX, is_fp32_dest_acc_en, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void acosh_tile_init() {
    MATH(SFPU_UNARY_INIT_FN(acosh, ckernel::sfpu::init_inverse_hyperbolic, (APPROX, is_fp32_dest_acc_en)));
}

// clang-format off
/**
 * Performs element-wise computation of the inverse hyperbolic cosine operation on each element of a tile
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void acosh_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_acosh,
        (APPROX, is_fp32_dest_acc_en, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void tan_tile_init() { MATH(SFPU_UNARY_INIT_FN(tan, ckernel::sfpu::tangent_init, (APPROX))); }

// clang-format off
/**
 * Performs element-wise computation of the trigonometric tan operation on each element of a tile
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void tan_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_tangent,
        (APPROX, is_fp32_dest_acc_en, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void asinh_tile_init() {
    MATH(SFPU_UNARY_INIT_FN(asinh, ckernel::sfpu::init_inverse_hyperbolic, (APPROX, is_fp32_dest_acc_en)));
}

// clang-format off
/**
 * Performs element-wise computation of the inverse hyperbolic sine operation on each element of a tile
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void asinh_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_asinh,
        (APPROX, is_fp32_dest_acc_en, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void atanh_tile_init() { MATH(SFPU_UNARY_INIT_FN(atanh, ckernel::sfpu::init_atanh, (APPROX, is_fp32_dest_acc_en))); }

// clang-format off
/**
 * Performs element-wise computation of the inverse hyperbolic tangent operation on each element of a tile
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void atanh_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_atanh,
        (APPROX, is_fp32_dest_acc_en, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC));
}

// clang-format off
/**
 * Performs element-wise computation of arcsine on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void asin_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_asin,
        (APPROX, is_fp32_dest_acc_en, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void asin_tile_init() { MATH(SFPU_UNARY_INIT_FN(asin, sfpu::asin_acos_init, (is_fp32_dest_acc_en))); }

// clang-format off
/**
 * Performs element-wise computation of arctan on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void atan_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_atan,
        (APPROX, is_fp32_dest_acc_en, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void atan_tile_init() {
    MATH(SFPU_UNARY_INIT_FN(atan, sfpu::atan_init, (true /*APPROXIMATION_MODE*/, is_fp32_dest_acc_en)));
}

// clang-format off
/**
 * Performs element-wise computation of arcossine on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void acos_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_acos,
        (APPROX, is_fp32_dest_acc_en, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void acos_tile_init() { MATH(SFPU_UNARY_INIT_FN(acos, sfpu::asin_acos_init, (is_fp32_dest_acc_en))); }

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void cosh_tile_init() { MATH(SFPU_UNARY_INIT_FN(cosh, ckernel::sfpu::cosh_init, (APPROX, is_fp32_dest_acc_en))); }

// clang-format off
/**
 * Performs element-wise computation of the trigonometric hyperbolic cosine operation on each element of a tile
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void cosh_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_cosh,
        (APPROX, is_fp32_dest_acc_en, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sinh_tile_init() { MATH(SFPU_UNARY_INIT_FN(sinh, ckernel::sfpu::sinh_init, (APPROX, is_fp32_dest_acc_en))); }

// clang-format off
/**
 * Performs element-wise computation of the trigonometric hyperbolic sine operation on each element of a tile
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sinh_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_sinh,
        (APPROX, is_fp32_dest_acc_en, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC));
}

#if !defined(TT_POLY_LLK_DISABLE) && ((defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)) && \
                                      defined(TT_METAL_SFPU_SINGLE_TILE_DST) && TT_METAL_SFPU_SINGLE_TILE_DST == 1)
#define TT_POLY_ACOS_BF16_ROUTE_ACTIVE 1
#else
#define TT_POLY_ACOS_BF16_ROUTE_ACTIVE 0
#endif

/** Internal BF16 typed-compiler route; public callers retain the stock entry point. */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void acos_tt_poly_bf16_tile(uint32_t idst) {
#if !TT_POLY_ACOS_BF16_ROUTE_ACTIVE
    acos_tile<is_fp32_dest_acc_en>(idst);
#else
    if constexpr (is_fp32_dest_acc_en) {
        acos_tile<is_fp32_dest_acc_en>(idst);
    } else {
        if (idst != 0) {
            acos_tile_init<is_fp32_dest_acc_en>();
            acos_tile<is_fp32_dest_acc_en>(idst);
            return;
        }
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            is_fp32_dest_acc_en,
            calculate_acos_tt_poly_bf16,
            (32 /* ITERATIONS */),
            idst,
            VectorMode::None));
    }
#endif
}

/** Initialize the internal BF16 typed-compiler route. */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void acos_tt_poly_bf16_tile_init() {
    acos_tile_init<is_fp32_dest_acc_en>();
}

#undef TT_POLY_ACOS_BF16_ROUTE_ACTIVE

#if !defined(TT_POLY_LLK_DISABLE) &&                                                                                 \
    ((defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)) && defined(TT_METAL_SFPU_SINGLE_TILE_DST) &&                \
     TT_METAL_SFPU_SINGLE_TILE_DST == 1 && defined(TT_POLY_BF16_UNARY_CONTEXT) && TT_POLY_BF16_UNARY_CONTEXT == 1 && \
     defined(SFPU_OP_PROGRAM_INIT_0))
#define TT_POLY_ACOSH_BF16_ROUTE_ACTIVE 1
#else
#define TT_POLY_ACOSH_BF16_ROUTE_ACTIVE 0
#endif

/** Internal BF16 typed-compiler route; public callers retain the stock entry point. */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void acosh_tt_poly_bf16_tile(uint32_t idst) {
#if !TT_POLY_ACOSH_BF16_ROUTE_ACTIVE
    acosh_tile<is_fp32_dest_acc_en>(idst);
#else
    if constexpr (!(is_fp32_dest_acc_en)) {
        acosh_tile<is_fp32_dest_acc_en>(idst);
    } else {
        if (idst != 0) {
            acosh_tile_init<is_fp32_dest_acc_en>();
            acosh_tile<is_fp32_dest_acc_en>(idst);
            return;
        }
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            is_fp32_dest_acc_en,
            calculate_acosh_tt_poly_bf16,
            (32 /* ITERATIONS */),
            idst,
            VectorMode::None));
    }
#endif
}

/** Initialize the internal BF16 typed-compiler route. */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void acosh_tt_poly_bf16_tile_init() {
#if !TT_POLY_ACOSH_BF16_ROUTE_ACTIVE
    acosh_tile_init<is_fp32_dest_acc_en>();
#else
    if constexpr (!(is_fp32_dest_acc_en)) {
        acosh_tile_init<is_fp32_dest_acc_en>();
    }
#endif
}

/** Initialize the selected single-tile program once, before its tile loop. */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void acosh_tt_poly_bf16_program_init() {
#if TT_POLY_ACOSH_BF16_ROUTE_ACTIVE
    if constexpr (!(!(is_fp32_dest_acc_en))) {
        MATH(sfpu::init_acosh_tt_poly_bf16());
    }
#endif
}

#undef TT_POLY_ACOSH_BF16_ROUTE_ACTIVE

#if !defined(TT_POLY_LLK_DISABLE) && ((defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)) && \
                                      defined(TT_METAL_SFPU_SINGLE_TILE_DST) && TT_METAL_SFPU_SINGLE_TILE_DST == 1)
#define TT_POLY_ATANH_BF16_ROUTE_ACTIVE 1
#else
#define TT_POLY_ATANH_BF16_ROUTE_ACTIVE 0
#endif

/** Internal BF16 typed-compiler route; public callers retain the stock entry point. */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void atanh_tt_poly_bf16_tile(uint32_t idst) {
#if !TT_POLY_ATANH_BF16_ROUTE_ACTIVE
    atanh_tile<is_fp32_dest_acc_en>(idst);
#else
    if constexpr (is_fp32_dest_acc_en) {
        atanh_tile<is_fp32_dest_acc_en>(idst);
    } else {
        if (idst != 0) {
            atanh_tile_init<is_fp32_dest_acc_en>();
            atanh_tile<is_fp32_dest_acc_en>(idst);
            return;
        }
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            is_fp32_dest_acc_en,
            calculate_atanh_tt_poly_bf16,
            (32 /* ITERATIONS */),
            idst,
            VectorMode::None));
    }
#endif
}

/** Initialize the internal BF16 typed-compiler route. */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void atanh_tt_poly_bf16_tile_init() {
    atanh_tile_init<is_fp32_dest_acc_en>();
}

#undef TT_POLY_ATANH_BF16_ROUTE_ACTIVE

}  // namespace ckernel
