// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_exp.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */
template <bool fast_and_approx = false, bool approx = false, uint32_t scale = 0x3F800000>
ALWI void exp_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_exponential_init<approx, fast_and_approx, scale>()));
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
 * | Argument    | Description                                                                | Type     | Valid Range                                           | Required |
 * |-------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst        | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | vector_mode | Specifies the vector mode for computation (e.g., Row, Column). (default: VectorMode::RC) | int      | Subject to specific hardware/kernel limits          | False    |
 * | scale       | Scale factor to apply if `scale_en` is true. (default: 0x3F80, which is 1.0f in FP16b) | uint16_t | Valid FP16b representation                          | False    |
 */
// clang-format on
template <
    bool fast_approx = false,
    bool approx = false,
    bool scale_en = false,
    bool skip_positive_check = false,
    int iterations = 8>
ALWI void exp_tile(uint32_t idst, int vector_mode = (int)VectorMode::RC, uint16_t scale = 0x3F80) {
    MATH((llk_math_eltwise_unary_sfpu_exponential<approx, fast_approx, scale_en, skip_positive_check, iterations>(
        idst, vector_mode, iterations, scale)));
}

}  // namespace ckernel
