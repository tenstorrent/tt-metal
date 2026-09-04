// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_trigonometry.h"
#endif

namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sin_tile_init() {
    MATH((sfpu::Trigonometry<sfpu::TrigOp::Sine, APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

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
    MATH((sfpu::Trigonometry<sfpu::TrigOp::Sine, APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void cos_tile_init() {
    MATH((sfpu::Trigonometry<sfpu::TrigOp::Cosine, APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

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
    MATH((sfpu::Trigonometry<sfpu::TrigOp::Cosine, APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void acosh_tile_init() {
    MATH((sfpu::Trigonometry<sfpu::TrigOp::Acosh, APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
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
    MATH((sfpu::Trigonometry<sfpu::TrigOp::Acosh, APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC)));
}

#ifndef ARCH_QUASAR
// Quasar does not implement tan / asin / acos / atan / cosh / sinh, so these entry points
// are not available there.

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void tan_tile_init() {
    MATH((sfpu::Trigonometry<sfpu::TrigOp::Tan, APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

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
    MATH((sfpu::Trigonometry<sfpu::TrigOp::Tan, APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC)));
}
#endif  // !ARCH_QUASAR

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void asinh_tile_init() {
    MATH((sfpu::Trigonometry<sfpu::TrigOp::Asinh, APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
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
    MATH((sfpu::Trigonometry<sfpu::TrigOp::Asinh, APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void atanh_tile_init() {
    MATH((sfpu::Trigonometry<sfpu::TrigOp::Atanh, APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
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
ALWI void atanh_tile(uint32_t idst) {
    MATH((sfpu::Trigonometry<sfpu::TrigOp::Atanh, APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC)));
}

#ifndef ARCH_QUASAR
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
    MATH((sfpu::Trigonometry<sfpu::TrigOp::Asin, APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void asin_tile_init() {
    MATH((sfpu::Trigonometry<sfpu::TrigOp::Asin, APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
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
ALWI void atan_tile(uint32_t idst) {
    MATH((sfpu::Trigonometry<sfpu::TrigOp::Atan, APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void atan_tile_init() {
    MATH((sfpu::Trigonometry<sfpu::TrigOp::Atan, true /*APPROXIMATION_MODE*/, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              init()));
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
    MATH((sfpu::Trigonometry<sfpu::TrigOp::Acos, APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void acos_tile_init() {
    MATH((sfpu::Trigonometry<sfpu::TrigOp::Acos, APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void cosh_tile_init() {
    MATH((sfpu::Trigonometry<sfpu::TrigOp::Cosh, APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
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
ALWI void cosh_tile(uint32_t idst) {
    MATH((sfpu::Trigonometry<sfpu::TrigOp::Cosh, APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sinh_tile_init() {
    MATH((sfpu::Trigonometry<sfpu::TrigOp::Sinh, APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
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
ALWI void sinh_tile(uint32_t idst) {
    MATH((sfpu::Trigonometry<sfpu::TrigOp::Sinh, APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC)));
}
#endif  // !ARCH_QUASAR

}  // namespace ckernel
