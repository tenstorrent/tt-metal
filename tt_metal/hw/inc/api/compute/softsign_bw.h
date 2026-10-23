// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#include "llk_assert.h"

// Every architecture declares these. The generated kernel exists for Blackhole, where TT-NN runs it;
// elsewhere each call asserts.
#if defined(ARCH_BLACKHOLE)
#ifdef TRISC_MATH
#include "ckernel_sfpu_softsign_bw_bf16.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif
#endif

namespace ckernel {

MATH(static_assert(!DST_ACCUM_MODE, "softsign_bw_tile evaluates BF16 DEST"));

// clang-format off
/**
 * Computes the input gradient grad * f'(x) of softsign over one tile, in BF16 DEST.
 * DEST tile idst holds x and tile idst + 1 holds grad; the result replaces x.
 *
 * Return value: None
 *
 * | Argument | Description                                   | Type     | Valid Range                                          | Required |
 * |----------|-----------------------------------------------|----------|------------------------------------------------------|----------|
 * | idst     | Index of the DST tile holding x; grad is next | uint32_t | idst + 1 must be less than the DST register capacity | True     |
 */
// clang-format on
ALWI void softsign_bw_tile(uint32_t idst) {
#if defined(ARCH_BLACKHOLE)
    // SFPU_UNARY_CALL checks idst alone; the kernel uses DEST tiles up to idst + 1.
    MATH(LLK_ASSERT(
        (idst + 2 <= get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>()),
        "softsign_bw_tile needs DEST tiles idst to idst + 1"));
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_softsign_bw_bf16, (32), idst, VectorMode::None));
#else
    MATH(LLK_ASSERT(false, "softsign_bw_tile is not supported on this architecture"));
#endif
}

/**
 * Initializes softsign_bw_tile. Must be called before softsign_bw_tile.
 */
ALWI void softsign_bw_tile_init() {
#if defined(ARCH_BLACKHOLE)
    MATH(SFPU_UNARY_INIT(unused));
    MATH(sfpu::init_softsign_bw_bf16());
#else
    MATH(LLK_ASSERT(false, "softsign_bw_tile is not supported on this architecture"));
#endif
}

}  // namespace ckernel
