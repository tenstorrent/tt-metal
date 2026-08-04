// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_polygamma.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs the elementwise polygamma function ψ^(n)(x) using a finite-sum plus
 * Euler–Maclaurin tail approximation:
 *
 *   ψ^(n)(x) ≈ (-1)^(n+1) * n! * [ Σ_{k=0}^{5} 1 / (x + k)^(n+1)
 *                                  + R_EM(x, n; B₂, B₄, B₆) ],
 *
 * where R_EM is an Euler–Maclaurin remainder using Bernoulli numbers B₂, B₄, and B₆
 * to approximate the infinite tail Σ_{k=6}^{∞} 1/(x+k)^(n+1), rather than performing
 * a hard truncation at k = 5. The kernel is intended for positive real inputs x and
 * integer orders 1 ≤ n ≤ 11.
 *
 * polygamma_tile(idst, n_packed, scale_packed);
 *
 * | Argument     | Description                                              | Type     | Valid Range                | Required |
 * |--------------|----------------------------------------------------------|----------|----------------------------|----------|
 * | idst         | Destination index                                        | uint32_t | 0 to (num_dests-1)         | Yes      |
 * | n_packed     | Order n as bit-cast uint32_t (from float). The value     | uint32_t | 1-11 (integer, as float    | Yes      |
 * |              | must encode an integer n in [1, 11]; other values        |          | bit pattern)               |          |
 * |              | are implementation-defined.                               |          |                            |          |
 * | scale_packed | Precomputed scaling factor (-1)^(n+1) * n! as bit-cast   | uint32_t | float bit pattern          | Yes      |
 * |              | uint32_t.                                                |          |                            |          |
 */
// clang-format on
ALWI void polygamma_tile(std::uint32_t idst, std::uint32_t n_packed, std::uint32_t scale_packed) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_polygamma,
        (APPROX, DST_ACCUM_MODE),
        idst,
        VectorMode::RC,
        n_packed,
        scale_packed));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void polygamma_tile_init() { MATH(SFPU_UNARY_INIT_FN(polygamma, sfpu::polygamma_init, (APPROX, DST_ACCUM_MODE))); }

}  // namespace ckernel
