// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
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
#ifdef ARCH_QUASAR
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _relu_min_,
        (SFPU_ITERATIONS /*ITERATIONS*/),
        idst,
        VectorMode::RC,
        0 /*threshold*/));
#else
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _relu_min_,
        (sfpi::vFloat /*VectorType*/, APPROX /*APPROXIMATION_MODE*/, 8 /*ITERATIONS*/, uint32_t /*T*/),
        idst,
        VectorMode::RC,
        0 /*threshold*/));
#endif
}
#ifndef ARCH_QUASAR
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
        _relu_max_,
        (sfpi::vInt /*VectorType*/, APPROX /*APPROXIMATION_MODE*/, 8 /*ITERATIONS*/, uint32_t /*T*/),
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

ALWI void relu_max_tile_init() { MATH(SFPU_UNARY_INIT(relu_max)); }
ALWI void relu_max_tile_init_pack() { PACK(SFPU_UNARY_INIT(relu_max)); }

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

ALWI void relu_min_tile_int32(uint32_t idst, uint32_t param0) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _relu_min_,
        (sfpi::vInt /*VectorType*/, APPROX /*APPROXIMATION_MODE*/, 8 /*ITERATIONS*/, uint32_t /*T*/),
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

ALWI void relu_min_tile_init() { MATH(SFPU_UNARY_INIT(relu_min)); }

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
}  // namespace ckernel
