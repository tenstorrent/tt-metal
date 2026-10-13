// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_defs.h"

namespace ckernel
{

constexpr std::uint32_t FAST_UNTILIZE_MAX_UNIT_DIM = 4;
constexpr std::uint32_t FAST_UNTILIZE_NUM_FACES    = 4;
// A 16-bit DEST half (512 rows) holds an eight-tile chunk; a 32-bit DEST half holds four tiles.
constexpr std::uint32_t FAST_UNTILIZE_MAX_UNIT_DIM_16BIT_DEST = 8;

// Shared math/pack DEST layout constants. One face is 16 DEST rows, and each
// tile contributes two 16-row face-pair strips: F0/F1 and F2/F3.
constexpr std::uint32_t FAST_UNTILIZE_PHASE_ROWS        = FACE_R_DIM;
constexpr std::uint32_t FAST_UNTILIZE_TILE_STRIDE_ROWS  = 2 * FAST_UNTILIZE_PHASE_ROWS;
constexpr std::uint32_t FAST_UNTILIZE_BLOCK_STRIDE_ROWS = FAST_UNTILIZE_MAX_UNIT_DIM * FAST_UNTILIZE_PHASE_ROWS;

// Math places all top face-pair rows first, followed by all bottom face-pair
// rows. The packer uses this same separation as the Z/W source stride.
constexpr std::uint32_t FAST_UNTILIZE_PHASE_PAIR_STRIDE_ROWS  = 2 * FAST_UNTILIZE_BLOCK_STRIDE_ROWS;
constexpr std::uint32_t FAST_UNTILIZE_TOP_STRIP_ROW_OFFSET    = 0;
constexpr std::uint32_t FAST_UNTILIZE_BOTTOM_STRIP_ROW_OFFSET = FAST_UNTILIZE_PHASE_PAIR_STRIDE_ROWS;

// BH pack phase selection is a DEST-target remap, not a physical DEST row
// index. These offsets intentionally do not mirror the math row offsets above:
// offset 128 exposes the top strip, while offset 0 exposes the bottom strip.
constexpr std::uint32_t FAST_UNTILIZE_PACK_TOP_STRIP_DEST_TARGET_OFFSET    = FAST_UNTILIZE_PHASE_PAIR_STRIDE_ROWS;
constexpr std::uint32_t FAST_UNTILIZE_PACK_BOTTOM_STRIP_DEST_TARGET_OFFSET = 0;

// Fast-untilize owns a private half-sync DEST region so each <=4-tile chunk can
// double-buffer independently of the ambient kernel sync mode.
constexpr DstSync FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE = DstSync::SyncHalf;

// Rows wider than eight tiles with a 16-bit DEST go in face-pair passes: one DEST half takes F0/F1 (then F2/F3) of up to
// sixteen tiles, so each pass packs sixteen output rows of the whole unit.
constexpr std::uint32_t FAST_UNTILIZE_MAX_PASS_UNIT_DIM = 16;

template <std::uint32_t full_ct_dim, bool is_fp32_dest_acc_en>
constexpr bool fast_untilize_face_passes()
{
    return !is_fp32_dest_acc_en && full_ct_dim > FAST_UNTILIZE_MAX_UNIT_DIM_16BIT_DEST;
}

// Sixteen-tile pass units, then the rest as a pass unit (9 to 15) or a chunk (2 to 8, too few to repay a second DEST
// hand-off); 17 left take a chunk of 8 first so no unit of 1 remains.
constexpr std::uint32_t fast_untilize_next_pass_row_unit(const std::uint32_t remaining_tiles)
{
    return remaining_tiles == FAST_UNTILIZE_MAX_PASS_UNIT_DIM + 1 ? FAST_UNTILIZE_MAX_UNIT_DIM_16BIT_DEST
           : remaining_tiles > FAST_UNTILIZE_MAX_PASS_UNIT_DIM    ? FAST_UNTILIZE_MAX_PASS_UNIT_DIM
                                                                  : remaining_tiles;
}

constexpr bool fast_untilize_unit_uses_passes(const std::uint32_t unit_dim)
{
    return unit_dim > FAST_UNTILIZE_MAX_UNIT_DIM_16BIT_DEST;
}

// MOP key of a pass unit, so a switch between pass and chunk units in one row reprograms the MOP.
constexpr std::uint32_t FAST_UNTILIZE_PASS_MOP_KEY = 0x100;

template <std::uint32_t full_ct_dim, bool is_fp32_dest_acc_en>
constexpr std::uint32_t fast_untilize_max_unit_dim()
{
    return (is_fp32_dest_acc_en || full_ct_dim <= FAST_UNTILIZE_MAX_UNIT_DIM) ? FAST_UNTILIZE_MAX_UNIT_DIM : FAST_UNTILIZE_MAX_UNIT_DIM_16BIT_DEST;
}

constexpr std::uint32_t fast_untilize_strip_rows(const std::uint32_t max_unit_dim)
{
    return (max_unit_dim > FAST_UNTILIZE_MAX_UNIT_DIM ? FAST_UNTILIZE_MAX_UNIT_DIM_16BIT_DEST : FAST_UNTILIZE_MAX_UNIT_DIM) * FAST_UNTILIZE_TILE_STRIDE_ROWS;
}

template <std::uint32_t max_unit_dim = FAST_UNTILIZE_MAX_UNIT_DIM>
constexpr std::uint32_t fast_untilize_next_unit_dim(const std::uint32_t remaining_tiles)
{
    // Avoid a trailing unit_dim=1, which the fast pack/math path does not support.
    // A tail of max_unit_dim + 1 tiles is decomposed as 2 + 3 (four-tile chunks) or 4 + 5 (eight-tile chunks).
    return (remaining_tiles > max_unit_dim + 1) ? max_unit_dim : (remaining_tiles == max_unit_dim + 1) ? (max_unit_dim + 1) / 2 : remaining_tiles;
}

template <std::uint32_t max_unit_dim = FAST_UNTILIZE_MAX_UNIT_DIM>
inline std::uint32_t fast_untilize_decompose_row(const std::uint32_t ct_dim, std::uint32_t* const unit_dims)
{
    std::uint32_t idx = 0;
    for (std::uint32_t remaining = ct_dim; remaining > 0;)
    {
        const std::uint32_t unit_dim = fast_untilize_next_unit_dim<max_unit_dim>(remaining);
        unit_dims[idx++]             = unit_dim;
        remaining -= unit_dim;
    }
    return idx;
}

} // namespace ckernel
