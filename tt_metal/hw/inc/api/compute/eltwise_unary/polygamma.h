// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_sfpu_polygamma.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs elementwise polygamma function: ψ^(n)(x) = (-1)^(n+1) * n! * Σ_{k=0}^{10} 1/(x+k)^(n+1)
 *
 * polygamma_tile(idst, n_packed, scale_packed);
 *
 * | Argument     | Description                                              | Type     | Valid Range                | Required |
 * |--------------|----------------------------------------------------------|----------|----------------------------|----------|
 * | idst         | Destination index                                        | uint32_t | 0 to (num_dests-1)         | Yes      |
 * | n_packed     | Order n as bit-cast uint32_t (from float)                | uint32_t | 1-10 (as float bits)       | Yes      |
 * | scale_packed | Precomputed (-1)^(n+1)*n! as bit-cast uint32_t           | uint32_t | float bits                 | Yes      |
 */
// clang-format on
ALWI void polygamma_tile(uint32_t idst, uint32_t n_packed, uint32_t scale_packed) {
    MATH((llk_math_eltwise_unary_sfpu_polygamma<APPROX, DST_ACCUM_MODE>(idst, n_packed, scale_packed)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void polygamma_tile_init() { MATH((llk_math_eltwise_unary_sfpu_polygamma_init<APPROX, DST_ACCUM_MODE>())); }

}  // namespace ckernel
