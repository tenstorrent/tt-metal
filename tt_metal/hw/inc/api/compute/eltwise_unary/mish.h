// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_mish.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs the elementwise mish activation:  mish(x) = x * tanh(softplus(x)).
 *
 * Implemented as a single fused SFPU kernel via the algebraic identity
 *   mish(x) = x * u (u + 2) / (u^2 + 2u + 2),  where u = exp(x).
 * This eliminates the explicit dependency on ops like tanh and softplus.
 *
 * For x >= 8, the kernel short-circuits to mish(x) = x.
 *
 * mish_tile<APPROXIMATION_MODE>(idst);
 *
 * | Argument | Description                                          | Type     | Valid Range                | Required |
 * |----------|------------------------------------------------------|----------|----------------------------|----------|
 * | idst     | Destination tile index                               | uint32_t | 0 to (num_dests-1)         | Yes      |
 */
// clang-format on
template <bool APPROXIMATION_MODE>
ALWI void mish_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_mish<APPROXIMATION_MODE, DST_ACCUM_MODE>(idst)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool APPROXIMATION_MODE>
ALWI void mish_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_mish_init<APPROXIMATION_MODE, DST_ACCUM_MODE>()));
}

}  // namespace ckernel
