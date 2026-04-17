// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_tanh_derivative.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */
template <bool fast_and_approx = false>
ALWI void tanh_derivative_tile_init() {
    MATH(SFPU_INIT_KERNEL_CALL(tanh_derivative, sfpu::tanh_derivative_sech2_init, fast_and_approx));
}

// clang-format off
/**
 * Performs element-wise computation of tanh derivative (sech²) on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Computes sech²(x) = 4·exp(-2|x|) / (1 + exp(-2|x|))² using accurate exp and reciprocal.
 * Avoids the catastrophic cancellation in the naive 1 - tanh²(x) formula.
 *
 * Return value: None
 *
 * | Argument         | Description                                                                | Type     | Valid Range                                           | Required |
 * |------------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index       | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | fast_and_approx  | Computation to be done faster and approximate                              | bool     |                                                       | False    |
 */
// clang-format on
template <bool fast_and_approx = false>
ALWI void tanh_derivative_tile(uint32_t idst) {
    MATH(SFPU_TWO_PARAM_KERNEL(
        calculate_tanh_derivative_sech2, fast_and_approx, DST_ACCUM_MODE, idst, (int)VectorMode::RC));
}

}  // namespace ckernel
