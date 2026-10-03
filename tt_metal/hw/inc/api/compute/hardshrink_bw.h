// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_hardshrink_bw_bf16.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

MATH(static_assert(!DST_ACCUM_MODE, "hardshrink_bw_tile evaluates BF16 DEST"));

// clang-format off
/**
 * Computes the input gradient grad * f'(x) of hardshrink over one tile, in BF16 DEST.
 * DEST tile idst holds x and tile idst + 1 holds grad; the result replaces x.
 *
 * Return value: None
 *
 * | Argument | Description                                   | Type     | Valid Range                                          | Required |
 * |----------|-----------------------------------------------|----------|------------------------------------------------------|----------|
 * | idst     | Index of the DST tile holding x; grad is next | uint32_t | idst + 1 must be less than the DST register capacity | True     |
 */
// clang-format on
ALWI void hardshrink_bw_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_hardshrink_bw_bf16, (32), idst, VectorMode::None));
}

/**
 * Initializes hardshrink_bw_tile. Must be called before hardshrink_bw_tile.
 */
ALWI void hardshrink_bw_tile_init() { MATH(SFPU_UNARY_INIT(unused)); }

}  // namespace ckernel
