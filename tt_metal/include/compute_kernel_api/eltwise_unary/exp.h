// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_exp.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

/**
 * Please refer to documentation for any_init.
 *
 * Template scale parameter is used when aprox is true and exp_tile is called with scale_en set to
 * true.
 *
 */
template <bool approx = false, uint32_t scale = 0x3F800000>
ALWI void exp_tile_init() {
    MATH(SFPU_TWO_TEMPLATE_PARAM_INIT(exponential, sfpu::exp_init, approx, scale));
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
 * | skip_positive_check     | Skip large-positive input check                                | bool     | true, false      | false   |
 * | iterations              | Number of iterations over 32-SFPU lanes to run                 | int      | Positive integer | 8       |
 *
 * | Argument    | Description                                                                | Type     | Valid Range                                           | Required |
 * |-------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst        | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | vector_mode | Specifies the vector mode for computation (default: VectorMode::RC)        | int      | Subject to specific hardware/kernel limits            | False    |
 * | scale       | Scale factor to apply in approximate or non-approximate mode if scale_en is true (default: 0x3F80, 1.0f in FP16b) | uint16_t | Valid FP16b representation                            | False    |
 */
// clang-format on
template <bool approx = false, bool scale_en = false, bool skip_positive_check = false, int iterations = 8>
ALWI void exp_tile(uint32_t idst, int vector_mode = (int)VectorMode::RC, uint16_t scale = p_sfpu::kCONST_1_FP16B) {
    MATH(SFPU_TEMPLATE_PARAMS_KERNEL(
        exponential, approx, DST_ACCUM_MODE, scale_en, skip_positive_check, iterations, idst, vector_mode, scale));
}

/**
 * Please refer to documentation for any_init.
 *
 * This init function is used for exponentialpiecewise approximation mode.
 *
 */
template <bool approx = false>
ALWI void exp_tile_piecewise_init() {
    MATH(SFPU_INIT_KERNEL_CALL(exponential, sfpu::exp_piecewise_init, approx));
}

/**
 * Please refer to documentation for any_init.
 *
 * This init function is used for SDPA first column computation.
 *
 */
template <bool approx = false>
ALWI void exp_tile_sdpa_first_column_init() {
    MATH(SFPU_INIT_KERNEL_CALL(exponential, sfpu::exp_sdpa_first_column_init, approx));
}

}  // namespace ckernel
