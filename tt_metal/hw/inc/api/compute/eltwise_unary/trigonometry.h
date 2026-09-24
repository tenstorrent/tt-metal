// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_trigonometry.h"
#endif

namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */
ALWI void sin_tile_init() { MATH((sfpu::Sine<APPROX>::init())); }

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
ALWI void sin_tile(std::uint32_t idst) {
    MATH((sfpu::Sine<APPROX, is_fp32_dest_acc_en, 8 /* ITERATIONS */>::run(idst)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void cos_tile_init() { MATH((sfpu::Cosine<APPROX>::init())); }

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
ALWI void cos_tile(std::uint32_t idst) {
    MATH((sfpu::Cosine<APPROX, is_fp32_dest_acc_en, 8 /* ITERATIONS */>::run(idst)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void acosh_tile_init() {
    MATH((sfpu::Acosh<APPROX, is_fp32_dest_acc_en>::init()));
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
ALWI void acosh_tile(std::uint32_t idst) {
    MATH((sfpu::Acosh<APPROX, is_fp32_dest_acc_en, 8 /* ITERATIONS */>::run(idst)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void tan_tile_init() { MATH((sfpu::Tangent<APPROX>::init())); }

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
ALWI void tan_tile(std::uint32_t idst) {
    MATH((sfpu::Tangent<APPROX, is_fp32_dest_acc_en, 8 /* ITERATIONS */>::run(idst)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void asinh_tile_init() {
    MATH((sfpu::Asinh<APPROX, is_fp32_dest_acc_en>::init()));
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
ALWI void asinh_tile(std::uint32_t idst) {
    MATH((sfpu::Asinh<APPROX, is_fp32_dest_acc_en, 8 /* ITERATIONS */>::run(idst)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void atanh_tile_init() {
    MATH((sfpu::Atanh<APPROX, is_fp32_dest_acc_en>::init()));
}

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
ALWI void atanh_tile(std::uint32_t idst) {
    MATH((sfpu::Atanh<APPROX, is_fp32_dest_acc_en, 8 /* ITERATIONS */>::run(idst)));
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
ALWI void asin_tile(std::uint32_t idst) {
    MATH((sfpu::Asin<APPROX, is_fp32_dest_acc_en, 8 /* ITERATIONS */>::run(idst)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void asin_tile_init() {
    MATH((sfpu::Asin<APPROX, is_fp32_dest_acc_en>::init()));
}

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
ALWI void atan_tile(std::uint32_t idst) {
    MATH((sfpu::Atan<APPROX, is_fp32_dest_acc_en, 8 /* ITERATIONS */>::run(idst)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void atan_tile_init() {
    MATH((sfpu::Atan<APPROX, is_fp32_dest_acc_en>::init()));
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
ALWI void acos_tile(std::uint32_t idst) {
    MATH((sfpu::Acos<APPROX, is_fp32_dest_acc_en, 8 /* ITERATIONS */>::run(idst)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void acos_tile_init() {
    MATH((sfpu::Acos<APPROX, is_fp32_dest_acc_en>::init()));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void cosh_tile_init() {
    MATH((sfpu::Cosh<APPROX, is_fp32_dest_acc_en>::init()));
}

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
ALWI void cosh_tile(std::uint32_t idst) {
    MATH((sfpu::Cosh<APPROX, is_fp32_dest_acc_en, 8 /* ITERATIONS */>::run(idst)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sinh_tile_init() {
    MATH((sfpu::Sinh<APPROX, is_fp32_dest_acc_en>::init()));
}

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
ALWI void sinh_tile(std::uint32_t idst) {
    MATH((sfpu::Sinh<APPROX, is_fp32_dest_acc_en, 8 /* ITERATIONS */>::run(idst)));
}

}  // namespace ckernel
