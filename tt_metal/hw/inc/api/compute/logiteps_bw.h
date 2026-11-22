// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#include "llk_assert.h"

// Every architecture declares these. The generated kernel exists for Blackhole and Wormhole, where TT-NN runs it;
// elsewhere a call fails to build.
#if defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)
#ifdef TRISC_MATH
#include "ckernel_sfpu_logiteps_bw_bf16.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif
#endif

namespace ckernel {

// clang-format off
/**
 * Computes the input gradient grad * f'(x) of logiteps over one tile, in BF16 DEST.
 * DEST tile idst holds x and tile idst + 1 holds grad; the result replaces x, and tile idst + 1 is consumed: its
 * contents afterwards are unspecified.
 *
 * Return value: None
 *
 * | Argument | Description                                   | Type     | Valid Range                                          | Required |
 * |----------|-----------------------------------------------|----------|------------------------------------------------------|----------|
 * | idst     | Index of the DST tile holding x; grad is next | uint32_t | idst + 1 must be less than the DST register capacity | True     |
 */
// clang-format on
template <bool on_board = false, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void logiteps_bw_tile([[maybe_unused]] uint32_t idst) {
#if defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)
    // Dependent on the template parameter, so only a call fails to build in an FP32 DEST kernel, not an include
    // (#61751 review).
    MATH(static_assert(!is_fp32_dest_acc_en, "logiteps_bw_tile evaluates BF16 DEST"));
    // SFPU_UNARY_CALL checks idst alone; the kernel uses DEST tiles up to idst + 1.
    MATH(LLK_ASSERT(
        (idst + 2 <= get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>()),
        "logiteps_bw_tile needs DEST tiles idst to idst + 1"));
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_logiteps_bw_bf16, (32 /*ITERATIONS*/), idst, VectorMode::None));
#else
    static_assert(on_board, "logiteps_bw_tile: Blackhole and Wormhole only");
#endif
}

/**
 * Initializes logiteps_bw_tile. Must be called before logiteps_bw_tile.
 */
template <bool on_board = false>
ALWI void logiteps_bw_tile_init() {
#if defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)
    MATH(SFPU_UNARY_INIT(unused));
#else
    static_assert(on_board, "logiteps_bw_tile: Blackhole and Wormhole only");
#endif
}

}  // namespace ckernel
