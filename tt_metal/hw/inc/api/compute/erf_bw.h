// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_erf_bw_bf16.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

MATH(static_assert(!DST_ACCUM_MODE, "erf_bw_tile evaluates BF16 DEST"));

// clang-format off
/**
 * Computes the input gradient grad * f'(x) of erf over one tile, in BF16 DEST.
 * DEST tile idst holds x and tile idst + 1 holds grad; the result replaces x. The kernel
 * also uses tile idst + 2 as scratch.
 *
 * Return value: None
 *
 * | Argument | Description                                   | Type     | Valid Range                                          | Required |
 * |----------|-----------------------------------------------|----------|------------------------------------------------------|----------|
 * | idst     | Index of the DST tile holding x; grad is next | uint32_t | idst + 2 must be less than the DST register capacity | True     |
 */
// clang-format on
ALWI void erf_bw_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_erf_bw_bf16, (32), idst, VectorMode::None));
    MATH(if constexpr (ckernel::sfpu::ErfBwBf16Config::needs_gradient) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_erf_bw_gradient_bf16, (32), idst, VectorMode::None));
    });
}

/**
 * Initializes erf_bw_tile. Must be called before erf_bw_tile.
 */
ALWI void erf_bw_tile_init() {
    MATH(SFPU_UNARY_INIT(unused));
    MATH(sfpu::init_erf_bw_bf16());
}

}  // namespace ckernel
