// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"

// Blackhole and Wormhole only: ckernel_sfpu_relu6_bw_bf16.h exists under those ckernel trees.
// Quasar keeps the composite; the generated kernel is not built there.
#if defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)

#ifdef TRISC_MATH
#include "ckernel_sfpu_relu6_bw_bf16.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

MATH(static_assert(!DST_ACCUM_MODE, "relu6_bw_tile evaluates BF16 DEST"));

// clang-format off
/**
 * Computes the input gradient grad * f'(x) of relu6 over one tile, in BF16 DEST.
 * DEST tile idst holds x and tile idst + 1 holds grad; the result replaces x.
 *
 * Return value: None
 *
 * | Argument | Description                                   | Type     | Valid Range                                          | Required |
 * |----------|-----------------------------------------------|----------|------------------------------------------------------|----------|
 * | idst     | Index of the DST tile holding x; grad is next | uint32_t | idst + 1 must be less than the DST register capacity | True     |
 */
// clang-format on
ALWI void relu6_bw_tile(uint32_t idst) {
    // SFPU_UNARY_CALL checks idst alone; the kernel uses DEST tiles up to idst + 1.
    MATH(LLK_ASSERT(
        (idst + 2 <= get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>()),
        "relu6_bw_tile needs DEST tiles idst to idst + 1"));
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_relu6_bw_bf16, (32), idst, VectorMode::None));
}

/**
 * Initializes relu6_bw_tile. Must be called before relu6_bw_tile.
 */
ALWI void relu6_bw_tile_init() { MATH(SFPU_UNARY_INIT(unused)); }

}  // namespace ckernel

#endif  // ARCH_BLACKHOLE || ARCH_WORMHOLE
