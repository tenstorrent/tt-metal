// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#include "tensor_shape.h"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "ckernel_sfpu_exp.h"
#endif

namespace ckernel {
/**
 * Controls whether the fast approximate exponential clamps very negative inputs.
 *
 * ClampToNegative (default): Inputs below ~-88.5 are clamped to -88.5. Safer but slightly slower.
 * None: No input clamping. Faster, but inputs below ~-88.5 will produce incorrect outputs. They
 *     will be guaranteed to be negative, so consider enabling packer ReLU when using this mode.
 */
enum class InputClamping : std::uint8_t {
    ClampToNegative = 1,
    None = 0,
};

/**
 * Please refer to documentation for any_init.
 *
 * Template scale parameter is used when approx is true and exp_tile is called with scale_en set to
 * true.
 *
 */
template <
    bool approx = false,
    std::uint32_t scale = 0x3F800000,
    InputClamping input_clamping = InputClamping::ClampToNegative,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void exp_tile_init() {
    MATH((sfpu::Exp<
          approx,
          is_fp32_dest_acc_en,
          false /* SCALE_EN */,
          8 /* ITERATIONS */,
          (input_clamping == InputClamping::ClampToNegative),
          scale>::init()));
}

// clang-format off
/**
 * Performs element-wise computation of exponential on each element of a tile
 * in the DST register. The DST register buffer must be in an
 * acquired state via an *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * The TENSOR_SHAPE template parameter selects the tile to process, e.g.
 * tensor_shape_from_tile_dims(32, 16) for the left column of faces; the default is the full 32x32 tile.
 *
 * Return value: None
 *
 * | Template Parameter      | Description                                                    | Type     | Valid Range      | Default |
 * |-------------------------|----------------------------------------------------------------|----------|------------------|---------|
 * | approx                  | Enable approximate mode.                                       | bool     | true, false      | false   |
 * | scale_en                | Enable input scaling by a constant factor in approximate or non-approximate mode | bool     | true, false      | false   |
 * | input_clamping          | If approx, controls whether very negative inputs are clamped to prevent incorrect outputs | InputClamping | ClampToNegative, None | ClampToNegative |
 * | iterations              | Number of iterations over 32-SFPU lanes to run                 | int      | Positive integer | 8       |
 *
 * | Argument    | Description                                                                | Type     | Valid Range                                           | Required |
 * |-------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst        | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | scale       | Scale factor to apply in approximate or non-approximate mode if scale_en is true (default: 0x3F80, 1.0f in FP16b) | uint16_t | Valid FP16b representation                            | False    |
 */
// clang-format on
template <
    bool approx = false,
    bool scale_en = false,
    InputClamping input_clamping = InputClamping::ClampToNegative,
    int iterations = 8,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    TensorShape TENSOR_SHAPE = DEFAULT_TENSOR_SHAPE>
ALWI void exp_tile(std::uint32_t idst, std::uint16_t scale = p_sfpu::kCONST_1_FP16B) {
    [[maybe_unused]] constexpr bool clamp_negative = input_clamping == InputClamping::ClampToNegative;
    MATH((sfpu::Exp<approx, is_fp32_dest_acc_en, scale_en, iterations, clamp_negative>::template run<TENSOR_SHAPE>(
        idst, scale)));
}

/**
 * Legacy overload selecting the faces to process with a VectorMode. Prefer the TensorShape template
 * parameter of the overload above: VectorMode::R and VectorMode::C correspond to
 * tensor_shape_from_tile_dims(16, 32) and tensor_shape_from_tile_dims(32, 16).
 */
template <
    bool approx = false,
    bool scale_en = false,
    InputClamping input_clamping = InputClamping::ClampToNegative,
    int iterations = 8,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void exp_tile(std::uint32_t idst, VectorMode vector_mode, std::uint16_t scale = p_sfpu::kCONST_1_FP16B) {
    [[maybe_unused]] constexpr bool clamp_negative = input_clamping == InputClamping::ClampToNegative;
    MATH((sfpu::Exp<approx, is_fp32_dest_acc_en, scale_en, iterations, clamp_negative>::run_vector_mode(
        vector_mode, idst, scale)));
}

// The pack-thread variants are not available on Quasar.
#ifndef ARCH_QUASAR

/**
 * Pack-thread variant of exp_tile_init. Runs the init on the pack thread
 * to enable FPU/SFPU overlap with math-thread matmul operations.
 */
template <
    bool approx = false,
    std::uint32_t scale = 0x3F800000,
    InputClamping input_clamping = InputClamping::ClampToNegative,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void exp_packthread_tile_init() {
    PACK((sfpu::Exp<
          approx,
          is_fp32_dest_acc_en,
          false /* SCALE_EN */,
          8 /* ITERATIONS */,
          (input_clamping == InputClamping::ClampToNegative),
          scale>::init()));
}

/**
 * Pack-thread variant of exp_tile. Runs the exp computation on the pack thread
 * to enable FPU/SFPU overlap with math-thread matmul operations.
 */
template <
    bool approx = false,
    bool scale_en = false,
    InputClamping input_clamping = InputClamping::ClampToNegative,
    int iterations = 8,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    TensorShape TENSOR_SHAPE = DEFAULT_TENSOR_SHAPE>
ALWI void exp_packthread_tile(std::uint32_t idst, std::uint16_t scale = p_sfpu::kCONST_1_FP16B) {
    [[maybe_unused]] constexpr bool clamp_negative = input_clamping == InputClamping::ClampToNegative;
    PACK((sfpu::Exp<approx, is_fp32_dest_acc_en, scale_en, iterations, clamp_negative>::template run<TENSOR_SHAPE>(
        idst, scale)));
}

/**
 * Legacy overload of exp_packthread_tile selecting the faces to process with a VectorMode. Prefer the
 * TensorShape template parameter of the overload above.
 */
template <
    bool approx = false,
    bool scale_en = false,
    InputClamping input_clamping = InputClamping::ClampToNegative,
    int iterations = 8,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void exp_packthread_tile(
    std::uint32_t idst, VectorMode vector_mode, std::uint16_t scale = p_sfpu::kCONST_1_FP16B) {
    [[maybe_unused]] constexpr bool clamp_negative = input_clamping == InputClamping::ClampToNegative;
    PACK((sfpu::Exp<approx, is_fp32_dest_acc_en, scale_en, iterations, clamp_negative>::run_vector_mode(
        vector_mode, idst, scale)));
}
#endif
}  // namespace ckernel
