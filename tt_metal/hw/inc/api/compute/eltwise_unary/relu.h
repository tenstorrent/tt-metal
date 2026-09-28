// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "api/compute/pack.h"

#include "api/compute/common_globals.h"
#if defined(TRISC_PACK) && !defined(TT_POLY_LLK_DISABLE)
#if __has_include("ckernel_sfpu_relu_min_bf16.h")
#include "ckernel_sfpu_relu_min_bf16.h"
#endif
#endif
#if defined(TRISC_PACK) && !defined(TT_POLY_LLK_DISABLE)
#if __has_include("ckernel_sfpu_relu6_bf16.h")
#include "ckernel_sfpu_relu6_bf16.h"
#endif
#endif
#if defined(TRISC_PACK) && !defined(TT_POLY_LLK_DISABLE)
#if __has_include("ckernel_sfpu_relu_bf16.h")
#include "ckernel_sfpu_relu_bf16.h"
#endif
#endif
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "ckernel_sfpu_relu.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

ALWI void relu_tile_init() { MATH(SFPU_UNARY_INIT(relu_min)); }

// clang-format off
/**
 * Performs element-wise computation of relu(x) = (0 if x is negative else x) on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *tile_regs_acquire* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void relu_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _relu_min_,
        (sfpi::vFloat /*VectorType*/, APPROX /*APPROXIMATION_MODE*/, 8 /*ITERATIONS*/, uint32_t /*T*/),
        idst,
        VectorMode::RC,
        0 /*threshold*/));
}

// clang-format off
/**
 * Performs element-wise computation of relu min (relu(min(x, lower_limit))) on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *tile_regs_acquire* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | lower_limit    | Upper limit of relu_min                                                    | uint32_t | Greater than 0                                        | True     |
 */
// clang-format on
ALWI void relu_min_tile(uint32_t idst, uint32_t param0) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _relu_min_,
        (sfpi::vFloat /*VectorType*/, APPROX /*APPROXIMATION_MODE*/, 8 /*ITERATIONS*/, uint32_t /*T*/),
        idst,
        VectorMode::RC,
        param0 /*threshold*/));
}

ALWI void relu_min_tile_init() { MATH(SFPU_UNARY_INIT(relu_min)); }

// clang-format off
/**
 * Performs element-wise computation of relu max (relu(max(x, upper_limit))) on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *tile_regs_acquire* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | upper_limit    | Upper limit of relu_min                                                    | uint32_t | Greater than 0                                        | True     |
 */
// clang-format on
ALWI void relu_max_tile(uint32_t idst, uint32_t param0) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _relu_max_,
        (sfpi::vFloat /*VectorType*/, APPROX /*APPROXIMATION_MODE*/, 8 /*ITERATIONS*/, uint32_t /*T*/),
        idst,
        VectorMode::RC,
        param0 /*threshold*/));
}

ALWI void relu_max_tile_init() { MATH(SFPU_UNARY_INIT(relu_max)); }

#ifndef ARCH_QUASAR
ALWI void relu_max_tile_pack(uint32_t idst, uint32_t param0) {
    PACK(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _relu_max_,
        (sfpi::vFloat /*VectorType*/, APPROX /*APPROXIMATION_MODE*/, 8 /*ITERATIONS*/, uint32_t /*T*/),
        idst,
        VectorMode::RC,
        param0 /*threshold*/));
}

ALWI void relu_max_tile_int32(uint32_t idst, uint32_t param0) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        relu_clamp_int,
        (APPROX /*APPROXIMATION_MODE*/, false /*IS_LOWER_BOUND*/, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC,
        param0 /*threshold*/));
}

ALWI void relu_max_tile_uint32(uint32_t idst, uint32_t param0) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        relu_clamp_uint,
        (APPROX /*APPROXIMATION_MODE*/, false /*IS_LOWER_BOUND*/, DataFormat::UInt32, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC,
        param0 /*threshold*/));
}

ALWI void relu_max_tile_uint16(uint32_t idst, uint32_t param0) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        relu_clamp_uint,
        (APPROX /*APPROXIMATION_MODE*/, false /*IS_LOWER_BOUND*/, DataFormat::UInt16, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC,
        param0 /*threshold*/));
}

ALWI void relu_max_tile_init_pack() { PACK(SFPU_UNARY_INIT(relu_max)); }

ALWI void relu_min_tile_int32(uint32_t idst, uint32_t param0) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        relu_clamp_int,
        (APPROX /*APPROXIMATION_MODE*/, true /*IS_LOWER_BOUND*/, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC,
        param0 /*threshold*/));
}

ALWI void relu_min_tile_uint32(uint32_t idst, uint32_t param0) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        relu_clamp_uint,
        (APPROX /*APPROXIMATION_MODE*/, true /*IS_LOWER_BOUND*/, DataFormat::UInt32, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC,
        param0 /*threshold*/));
}

ALWI void relu_min_tile_uint16(uint32_t idst, uint32_t param0) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        relu_clamp_uint,
        (APPROX /*APPROXIMATION_MODE*/, true /*IS_LOWER_BOUND*/, DataFormat::UInt16, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC,
        param0 /*threshold*/));
}

ALWI void relu_tile_int32(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _relu_min_,
        (sfpi::vInt /*VectorType*/, APPROX /*APPROXIMATION_MODE*/, 8 /*ITERATIONS*/, uint32_t /*T*/),
        idst,
        VectorMode::RC,
        0 /*threshold*/));
}

// clang-format off
/**
 * Performs element-wise computation of leaky relu (relu(x) + slope*-relu(-x)) on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *tile_regs_acquire* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | slope          | slope used in leaky relu - will reinterpret unsigned int to float          | uint32_t | Greater than 0                                        | True     |
 */
// clang-format on
ALWI void leaky_relu_tile(uint32_t idst, uint32_t slope = 0) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_lrelu, (APPROX), idst, VectorMode::RC, slope));
}

ALWI void leaky_relu_tile_init() { MATH(SFPU_UNARY_INIT(lrelu)); }
#endif
#if !defined(TT_POLY_LLK_DISABLE) && \
    ((defined(TT_POLY_RELU_BF16_AVAILABLE)) && defined(TT_POLY_BF16_UNARY_CONTEXT) && TT_POLY_BF16_UNARY_CONTEXT == 1)
#define TT_POLY_RELU_BF16_ROUTE_ACTIVE 1
#else
#define TT_POLY_RELU_BF16_ROUTE_ACTIVE 0
#endif

/** Internal BF16 typed-compiler route; public callers retain the stock entry point. */
ALWI void relu_tt_poly_bf16_tile(uint32_t idst) {
#if !TT_POLY_RELU_BF16_ROUTE_ACTIVE
    relu_tile(idst);
#else
    if constexpr (DST_ACCUM_MODE) {
        relu_tile(idst);
    } else {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_relu_tt_poly_bf16, (8 /* ITERATIONS */), idst, VectorMode::RC));
    }
#endif
}

/** Initialize the internal BF16 typed-compiler route. */
ALWI void relu_tt_poly_bf16_tile_init() {
#if !TT_POLY_RELU_BF16_ROUTE_ACTIVE
    relu_tile_init();
#else
    if constexpr (DST_ACCUM_MODE) {
        relu_tile_init();
    } else {
        relu_tile_init();
        MATH(sfpu::init_relu_tt_poly_bf16());
    }
#endif
}

/** Arm the selected packer clamp once, before the tile loop. */
ALWI void relu_tt_poly_bf16_program_init() {
#if TT_POLY_RELU_BF16_ROUTE_ACTIVE && defined(TRISC_PACK)
    if constexpr (!(DST_ACCUM_MODE) && ttpoly_generated::ReluBf16Config::kRoute == 4u) {
        if constexpr (ttpoly_generated::ReluBf16Config::kHasUpper) {
            pack_relu_config(
                ckernel::ReluConfig::max_threshold(ttpoly_generated::ReluBf16Config::kPackReluThresholdBits));
        } else {
            pack_relu_config(ckernel::ReluConfig::zero());
        }
    }
#endif
}

/** Disarm the packer clamp after the final pack/release. */
ALWI void relu_tt_poly_bf16_program_finish() {
#if TT_POLY_RELU_BF16_ROUTE_ACTIVE && defined(TRISC_PACK)
    if constexpr (!(DST_ACCUM_MODE) && ttpoly_generated::ReluBf16Config::kRoute == 4u) {
        pack_relu_config(ckernel::ReluConfig::none());
    }
#endif
}

#undef TT_POLY_RELU_BF16_ROUTE_ACTIVE

#if !defined(TT_POLY_LLK_DISABLE) && ((defined(TT_POLY_RELU6_BF16_AVAILABLE)) && \
                                      defined(TT_POLY_BF16_UNARY_CONTEXT) && TT_POLY_BF16_UNARY_CONTEXT == 1)
#define TT_POLY_RELU6_BF16_ROUTE_ACTIVE 1
#else
#define TT_POLY_RELU6_BF16_ROUTE_ACTIVE 0
#endif

/** Internal BF16 typed-compiler route; public callers retain the stock entry point. */
ALWI void relu6_tt_poly_bf16_tile(uint32_t idst) {
#if !TT_POLY_RELU6_BF16_ROUTE_ACTIVE
    relu_max_tile(idst, 0x40c00000u);
#else
    if constexpr (DST_ACCUM_MODE) {
        relu_max_tile(idst, 0x40c00000u);
    } else {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_relu6_tt_poly_bf16, (8 /* ITERATIONS */), idst, VectorMode::RC));
    }
#endif
}

/** Initialize the internal BF16 typed-compiler route. */
ALWI void relu6_tt_poly_bf16_tile_init() {
#if !TT_POLY_RELU6_BF16_ROUTE_ACTIVE
    relu_max_tile_init();
#else
    if constexpr (DST_ACCUM_MODE) {
        relu_max_tile_init();
    } else {
        relu_max_tile_init();
        MATH(sfpu::init_relu6_tt_poly_bf16());
    }
#endif
}

/** Arm the selected packer clamp once, before the tile loop. */
ALWI void relu6_tt_poly_bf16_program_init() {
#if TT_POLY_RELU6_BF16_ROUTE_ACTIVE && defined(TRISC_PACK)
    if constexpr (!(DST_ACCUM_MODE) && ttpoly_generated::Relu6Bf16Config::kRoute == 4u) {
        if constexpr (ttpoly_generated::Relu6Bf16Config::kHasUpper) {
            pack_relu_config(
                ckernel::ReluConfig::max_threshold(ttpoly_generated::Relu6Bf16Config::kPackReluThresholdBits));
        } else {
            pack_relu_config(ckernel::ReluConfig::zero());
        }
    }
#endif
}

/** Disarm the packer clamp after the final pack/release. */
ALWI void relu6_tt_poly_bf16_program_finish() {
#if TT_POLY_RELU6_BF16_ROUTE_ACTIVE && defined(TRISC_PACK)
    if constexpr (!(DST_ACCUM_MODE) && ttpoly_generated::Relu6Bf16Config::kRoute == 4u) {
        pack_relu_config(ckernel::ReluConfig::none());
    }
#endif
}

#undef TT_POLY_RELU6_BF16_ROUTE_ACTIVE

#if !defined(TT_POLY_LLK_DISABLE) && ((defined(TT_POLY_RELU_MIN_BF16_AVAILABLE)) && \
                                      defined(TT_POLY_BF16_UNARY_CONTEXT) && TT_POLY_BF16_UNARY_CONTEXT == 1)
#define TT_POLY_RELU_MIN_BF16_ROUTE_ACTIVE 1
#else
#define TT_POLY_RELU_MIN_BF16_ROUTE_ACTIVE 0
#endif

/** Internal BF16 typed-compiler route; public callers retain the stock entry point. */
ALWI void relu_min_tt_poly_bf16_tile(uint32_t idst, uint32_t param0) {
#if !TT_POLY_RELU_MIN_BF16_ROUTE_ACTIVE
    relu_min_tile(idst, param0);
#else
    if constexpr (DST_ACCUM_MODE) {
        relu_min_tile(idst, param0);
    } else {
        if (param0 != 0x00000000u) {
            relu_min_tile(idst, param0);
            return;
        }
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_relu_min_tt_poly_bf16,
            (8 /* ITERATIONS */),
            idst,
            VectorMode::RC));
    }
#endif
}

/** Initialize the internal BF16 typed-compiler route. */
ALWI void relu_min_tt_poly_bf16_tile_init() {
#if !TT_POLY_RELU_MIN_BF16_ROUTE_ACTIVE
    relu_min_tile_init();
#else
    if constexpr (DST_ACCUM_MODE) {
        relu_min_tile_init();
    } else {
        relu_min_tile_init();
        MATH(sfpu::init_relu_min_tt_poly_bf16());
    }
#endif
}

/** Arm the selected packer clamp once, before the tile loop. */
ALWI void relu_min_tt_poly_bf16_program_init() {
#if TT_POLY_RELU_MIN_BF16_ROUTE_ACTIVE && defined(TRISC_PACK)
    if constexpr (!(DST_ACCUM_MODE) && ttpoly_generated::ReluMinBf16Config::kRoute == 4u) {
        if constexpr (ttpoly_generated::ReluMinBf16Config::kHasUpper) {
            pack_relu_config(
                ckernel::ReluConfig::max_threshold(ttpoly_generated::ReluMinBf16Config::kPackReluThresholdBits));
        } else {
            pack_relu_config(ckernel::ReluConfig::zero());
        }
    }
#endif
}

/** Disarm the packer clamp after the final pack/release. */
ALWI void relu_min_tt_poly_bf16_program_finish() {
#if TT_POLY_RELU_MIN_BF16_ROUTE_ACTIVE && defined(TRISC_PACK)
    if constexpr (!(DST_ACCUM_MODE) && ttpoly_generated::ReluMinBf16Config::kRoute == 4u) {
        pack_relu_config(ckernel::ReluConfig::none());
    }
#endif
}

#undef TT_POLY_RELU_MIN_BF16_ROUTE_ACTIVE

}  // namespace ckernel
