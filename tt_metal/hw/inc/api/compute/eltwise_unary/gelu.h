// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "ckernel_sfpu_gelu.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {
/**
 * Please refer to documentation for any_init.
 */
template <bool fast_and_approx = true, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void gelu_tile_init() {
    MATH(SFPU_UNARY_INIT_FN(gelu, sfpu::gelu_init, (fast_and_approx, is_fp32_dest_acc_en)));
}

// clang-format off
/**
 * Performs element-wise computation of gelu  on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument         | Description                                                                | Type     | Valid Range                                           | Required |
 * |------------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index       | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | fast_and_approx  | Computation to be done faster and approximate                              | bool     |                                                       | False    |
 */
// clang-format on
template <bool fast_and_approx = true, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void gelu_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_gelu, (fast_and_approx, is_fp32_dest_acc_en), idst, VectorMode::RC));
}

#ifndef ARCH_QUASAR
template <bool fast_and_approx = true, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void gelu_tile_init_pack() {
    PACK(SFPU_UNARY_INIT_FN(gelu, sfpu::gelu_init, (fast_and_approx, is_fp32_dest_acc_en)));
}

template <bool fast_and_approx = true, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void gelu_tile_pack(uint32_t idst) {
    PACK(SFPU_UNARY_CALL(
        DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_gelu, (fast_and_approx, is_fp32_dest_acc_en), idst, VectorMode::RC));
}

/**
 * Init for gelu_tanh_tile. See gelu_tanh_tile() for semantics.
 */
ALWI void gelu_tanh_tile_init() {
    MATH(llk_math_eltwise_unary_sfpu_init<SfpuType::gelu_tanh>(sfpu::gelu_tanh_init));
}

ALWI void gelu_tanh_tile_init_pack() {
    PACK(llk_math_eltwise_unary_sfpu_init<SfpuType::gelu_tanh>(sfpu::gelu_tanh_init));
}

// clang-format off
/**
 * Element-wise GELU using the tanh approximation, computed in FP32:
 *   GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 *
 * Intended as a fused-activation drop-in for matmuls that need the
 * tanh-GELU (e.g. F.gelu(approximate="tanh")).
 *
 * Return value: None
 *
 * | Argument         | Description                                                                | Type     | Valid Range                                           | Required |
 * |------------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index       | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void gelu_tanh_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_gelu_tanh, (is_fp32_dest_acc_en), idst, VectorMode::RC));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void gelu_tanh_tile_pack(uint32_t idst) {
    PACK(SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_gelu_tanh, (is_fp32_dest_acc_en), idst, VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool fast_and_approx = false>
ALWI void gelu_derivative_tile_init() {
    MATH(SFPU_UNARY_INIT_FN(gelu_derivative, sfpu::gelu_derivative_polynomial_init, (fast_and_approx)));
}

// clang-format off
/**
 * Performs element-wise computation of GELU derivative on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * When fast_and_approx=false (default): uses piecewise polynomial approximation
 * with Max ULP = 1 across all BF16 inputs.
 *
 * When fast_and_approx=true: uses the same piecewise polynomial core but skips
 * the Mills-ratio correction in the negative tail (x < -3), trading ~1% relative
 * accuracy in that region for fewer operations.
 *
 * Return value: None
 *
 * | Argument         | Description                                                                | Type     | Valid Range                                           | Required |
 * |------------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index       | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | fast_and_approx  | Computation to be done faster and approximate                              | bool     |                                                       | False    |
 */
// clang-format on
template <bool fast_and_approx = false, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void gelu_derivative_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_gelu_derivative_polynomial,
        (fast_and_approx, is_fp32_dest_acc_en),
        idst,
        VectorMode::RC));
}

#endif
}  // namespace ckernel
