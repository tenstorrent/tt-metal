// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#if !defined(ARCH_BLACKHOLE)
#error "experimental/block_max8.h is supported on Blackhole only"
#else

#include "api/compute/common.h"
#ifdef TRISC_MATH
#include "experimental/llk_sfpu/llk_math_block_max8.h"
#endif

namespace ckernel {

/**
 * @brief Initialize Blackhole BF16 group-of-eight max pooling on the math thread.
 *
 * Call before block_max8. The SFPU reads DST directly and needs no unpack
 * operation. The caller owns DST acquire/commit/wait/release and CB publication.
 */
ALWI void block_max8_init() { MATH((llk_math_block_max8_init<DST_ACCUM_MODE>())); }

/**
 * @brief Replace one DST tile with 128 contiguous block maxima in place.
 *
 * The input is a logical 32x32 BF16 tile. Each group of eight consecutive scores
 * produces one maximum. Invalid scores are masked to negative infinity before
 * pooling. The first eight physical DST rows contain the result; the original
 * scores and the rest of the tile are not preserved. Pack using the existing
 * row-pack compute API configured for eight rows.
 * @param dst_index Runtime tile index in the acquired BF16 DST half.
 * @param valid_scores Valid row-major prefix length, from zero through 1024; defaults to the full tile.
 */
ALWI void block_max8(uint32_t dst_index, uint32_t valid_scores = TILE_R_DIM * TILE_C_DIM) {
    MATH((llk_math_block_max8<DST_ACCUM_MODE>(dst_index, valid_scores)));
}

}  // namespace ckernel

#endif
