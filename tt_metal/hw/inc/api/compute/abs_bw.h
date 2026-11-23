// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#include "llk_assert.h"

// Every architecture declares these. The generated kernel exists for Blackhole, Wormhole and Quasar (TT-NN runs it on
// Blackhole and Wormhole); elsewhere a call fails to build.
#if defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE) || defined(ARCH_QUASAR)
#ifdef TRISC_MATH
#include "ckernel_sfpu_abs_bw_bf16.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif
#endif

namespace ckernel {

MATH(static_assert(!DST_ACCUM_MODE, "abs_bw_tile evaluates BF16 DEST"));

// clang-format off
/**
 * Computes the input gradient grad * f'(x) of abs over one tile, in BF16 DEST.
 * DEST tile idst holds x and tile idst + 1 holds grad; the result replaces x.
 *
 * Return value: None
 *
 * | Argument | Description                                   | Type     | Valid Range                                          | Required |
 * |----------|-----------------------------------------------|----------|------------------------------------------------------|----------|
 * | idst     | Index of the DST tile holding x; grad is next | uint32_t | idst + 1 must be less than the DST register capacity | True     |
 */
// clang-format on
template <bool on_board = false>
ALWI void abs_bw_tile([[maybe_unused]] uint32_t idst) {
#if defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE) || defined(ARCH_QUASAR)
    // SFPU_UNARY_CALL checks idst alone; the kernel uses DEST tiles up to idst + 1.
#ifdef ARCH_QUASAR
    MATH(LLK_ASSERT(
        (idst + 2 <= get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()),
        "abs_bw_tile needs DEST tiles idst to idst + 1"));
#else
    MATH(LLK_ASSERT(
        (idst + 2 <= get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>()),
        "abs_bw_tile needs DEST tiles idst to idst + 1"));
#endif
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_abs_bw_bf16, (32 /*ITERATIONS*/), idst, VectorMode::None));
#else
    static_assert(on_board, "abs_bw_tile: Blackhole, Wormhole and Quasar only");
#endif
}

/**
 * Initializes abs_bw_tile. Must be called before abs_bw_tile.
 */
template <bool on_board = false>
ALWI void abs_bw_tile_init() {
#if defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE) || defined(ARCH_QUASAR)
    MATH(SFPU_UNARY_INIT(unused));
#else
    static_assert(on_board, "abs_bw_tile: Blackhole, Wormhole and Quasar only");
#endif
}

}  // namespace ckernel
