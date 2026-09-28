// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "ckernel_sfpu_exp.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {
/**
 * Controls whether the fast approximate exponential clamps very negative inputs.
 *
 * ClampToNegative (default): Inputs below ~-88.5 are clamped to -88.5. Safer but slightly slower.
 * None: No input clamping. Faster, but inputs below ~-88.5 will produce incorrect outputs. They
 *     will be guaranteed to be negative, so consider enabling packer ReLU when using this mode.
 */
enum class InputClamping : uint8_t {
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
    uint32_t scale = 0x3F800000,
    InputClamping input_clamping = InputClamping::ClampToNegative, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void exp_tile_init() {
    MATH(SFPU_UNARY_INIT_FN(
        exponential,
        sfpu::exp_init,
        (approx, scale, (input_clamping == InputClamping::ClampToNegative), is_fp32_dest_acc_en)));
}

// clang-format off
/**
 * Performs element-wise computation of exponential on each element of a tile
 * in the DST register. The DST register buffer must be in an
 * acquired state via an *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
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
 * | vector_mode | Specifies the vector mode for computation (default: VectorMode::RC)        | VectorMode | Subject to specific hardware/kernel limits            | False    |
 * | scale       | Scale factor to apply in approximate or non-approximate mode if scale_en is true (default: 0x3F80, 1.0f in FP16b) | uint16_t | Valid FP16b representation                            | False    |
 */
// clang-format on
template <
    bool approx = false,
    bool scale_en = false,
    InputClamping input_clamping = InputClamping::ClampToNegative,
    int iterations = 8, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void exp_tile(uint32_t idst, VectorMode vector_mode = VectorMode::RC, uint16_t scale = p_sfpu::kCONST_1_FP16B) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_exponential,
        (approx, is_fp32_dest_acc_en, scale_en, iterations, (input_clamping == InputClamping::ClampToNegative)),
        idst,
        vector_mode,
        scale));
}

#ifndef ARCH_QUASAR

/**
 * Pack-thread variant of exp_tile_init. Runs the init on the pack thread
 * to enable FPU/SFPU overlap with math-thread matmul operations.
 */
template <
    bool approx = false,
    uint32_t scale = 0x3F800000,
    InputClamping input_clamping = InputClamping::ClampToNegative, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void exp_packthread_tile_init() {
    PACK(llk_math_eltwise_unary_sfpu_init<SfpuType::exponential>(
        sfpu::exp_init<approx, scale, (input_clamping == InputClamping::ClampToNegative), is_fp32_dest_acc_en>));
}

/**
 * Pack-thread variant of exp_tile. Runs the exp computation on the pack thread
 * to enable FPU/SFPU overlap with math-thread matmul operations.
 */
template <
    bool approx = false,
    bool scale_en = false,
    InputClamping input_clamping = InputClamping::ClampToNegative,
    int iterations = 8, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void exp_packthread_tile(
    uint32_t idst, VectorMode vector_mode = VectorMode::RC, uint16_t scale = p_sfpu::kCONST_1_FP16B) {
    PACK(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_exponential,
        (approx, is_fp32_dest_acc_en, scale_en, iterations, (input_clamping == InputClamping::ClampToNegative)),
        idst,
        vector_mode,
        scale));
}
#endif
#if !defined(TT_POLY_LLK_DISABLE) &&                                                                                 \
    ((defined(TT_POLY_EXP_BF16_AVAILABLE)) && defined(TT_METAL_SFPU_SINGLE_TILE_DST) &&                              \
     TT_METAL_SFPU_SINGLE_TILE_DST == 1 && defined(TT_POLY_BF16_UNARY_CONTEXT) && TT_POLY_BF16_UNARY_CONTEXT == 1 && \
     defined(SFPU_OP_PROGRAM_INIT_0))
#define TT_POLY_EXP_BF16_ROUTE_ACTIVE 1
#else
#define TT_POLY_EXP_BF16_ROUTE_ACTIVE 0
#endif

/** Internal BF16 typed-compiler route; public callers retain the stock entry point. */
template <bool approx = false, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void exp_tt_poly_bf16_tile(uint32_t idst) {
#if !TT_POLY_EXP_BF16_ROUTE_ACTIVE
    exp_tile<approx, false, InputClamping::ClampToNegative, 8, is_fp32_dest_acc_en>(
        idst, VectorMode::RC, p_sfpu::kCONST_1_FP16B);
#else
    if constexpr (is_fp32_dest_acc_en) {
        exp_tile<approx, false, InputClamping::ClampToNegative, 8, is_fp32_dest_acc_en>(
            idst, VectorMode::RC, p_sfpu::kCONST_1_FP16B);
    } else {
        if (idst != 0) {
            exp_tile_init<approx, 0x3F800000, InputClamping::ClampToNegative, is_fp32_dest_acc_en>();
            exp_tile<approx, false, InputClamping::ClampToNegative, 8, is_fp32_dest_acc_en>(
                idst, VectorMode::RC, p_sfpu::kCONST_1_FP16B);
            MATH(SFPU_UNARY_INIT_FN(
                exponential,
                sfpu::init_exp_tt_poly_bf16,
                (approx,
                 0x3F800000,
                 (InputClamping::ClampToNegative == InputClamping::ClampToNegative),
                 is_fp32_dest_acc_en)));
            return;
        }
        if constexpr (approx) {
            exp_tile_init<approx, 0x3F800000, InputClamping::ClampToNegative, is_fp32_dest_acc_en>();
            exp_tile<approx, false, InputClamping::ClampToNegative, 8, is_fp32_dest_acc_en>(
                idst, VectorMode::RC, p_sfpu::kCONST_1_FP16B);
            MATH(SFPU_UNARY_INIT_FN(
                exponential,
                sfpu::init_exp_tt_poly_bf16,
                (false,
                 0x3F800000,
                 (InputClamping::ClampToNegative == InputClamping::ClampToNegative),
                 is_fp32_dest_acc_en)));
            return;
        }
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            is_fp32_dest_acc_en,
            calculate_exp_tt_poly_bf16,
            (32 /* ITERATIONS */),
            idst,
            VectorMode::None));
    }
#endif
}

/** Initialize the internal BF16 typed-compiler route. */
template <bool approx = false, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void exp_tt_poly_bf16_tile_init() {
    if constexpr (approx) {
        exp_tile_init<approx, 0x3F800000, InputClamping::ClampToNegative, is_fp32_dest_acc_en>();
        return;
    }

#if !TT_POLY_EXP_BF16_ROUTE_ACTIVE
    exp_tile_init<approx, 0x3F800000, InputClamping::ClampToNegative, is_fp32_dest_acc_en>();
#else
    if constexpr (is_fp32_dest_acc_en) {
        exp_tile_init<approx, 0x3F800000, InputClamping::ClampToNegative, is_fp32_dest_acc_en>();
    }
#endif
}

/** Initialize the selected single-tile program once, before its tile loop. */
template <bool approx = false, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void exp_tt_poly_bf16_program_init() {
#if TT_POLY_EXP_BF16_ROUTE_ACTIVE
    if constexpr (!(is_fp32_dest_acc_en)) {
        MATH(SFPU_UNARY_INIT_FN(
            exponential,
            sfpu::init_exp_tt_poly_bf16,
            (approx,
             0x3F800000,
             (InputClamping::ClampToNegative == InputClamping::ClampToNegative),
             is_fp32_dest_acc_en)));
    }
#endif
}

#undef TT_POLY_EXP_BF16_ROUTE_ACTIVE

}  // namespace ckernel
