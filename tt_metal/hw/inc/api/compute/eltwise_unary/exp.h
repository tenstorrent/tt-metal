// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
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
 * Template scale parameter is used when approx and fast_and_approx are true and exp_tile is called with scale_en set to
 * true.
 *
 */
template <
    bool approx = false,
    bool fast_and_approx = true,
    uint32_t scale = 0x3F800000,
    InputClamping input_clamping = InputClamping::ClampToNegative>
ALWI void exp_tile_init() {
    MATH(SFPU_TEMPLATE_INIT_KERNEL(
        exponential,
        sfpu::exp_init,
        approx,
        fast_and_approx,
        scale,
        (input_clamping == InputClamping::ClampToNegative)));
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
 * | fast_and_approx         | If approx is true, enable fast approximation.                  | bool     | true, false      | true   |
 * | scale_en                | Enable input scaling by a constant factor in approximate or non-approximate mode | bool     | true, false      | false   |
 * | skip_positive_check     | Skip large-positive input check                                | bool     | true, false      | false   |
 * | input_clamping          | If approx && fast_and_approx, controls whether very negative inputs are clamped to prevent incorrect outputs | InputClamping | ClampToNegative, None | ClampToNegative |
 * | iterations              | Number of iterations over 32-SFPU lanes to run                 | int      | Positive integer | 8       |
 *
 * | Argument    | Description                                                                | Type     | Valid Range                                           | Required |
 * |-------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst        | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | vector_mode | Specifies the vector mode for computation (default: VectorMode::RC)        | int      | Subject to specific hardware/kernel limits            | False    |
 * | scale       | Scale factor to apply in approximate or non-approximate mode if scale_en is true (default: 0x3F80, 1.0f in FP16b) | uint16_t | Valid FP16b representation                            | False    |
 */
// clang-format on
template <
    bool approx = false,
    bool fast_and_approx = true,
    bool scale_en = false,
    bool skip_positive_check = false,
    InputClamping input_clamping = InputClamping::ClampToNegative,
    int iterations = 8>
ALWI void exp_tile(uint32_t idst, int vector_mode = (int)VectorMode::RC, uint16_t scale = p_sfpu::kCONST_1_FP16B) {
    MATH(SFPU_TEMPLATE_PARAMS_KERNEL_FN(
        calculate_exponential,
        approx,
        fast_and_approx,
        DST_ACCUM_MODE,
        scale_en,
        skip_positive_check,
        (input_clamping == InputClamping::ClampToNegative),
        iterations,
        idst,
        vector_mode,
        scale));
}

}  // namespace ckernel
