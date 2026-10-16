// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Row-section padding: fill tile columns [PAD_COL_START, PAD_COL_END) of tile row 0 of a 16-bit 32x32
// tile with a 16-bit scalar. Tile row 0 is row 0 of face 0 (columns 0-15) followed by row 0 of face 1
// (columns 16-31), so the section splits into at most one contiguous segment per face.
// The section always runs to the end of the row. Keep PAD_COL_START in sync with the python test.
constexpr std::uint32_t TILE_COLS     = 32;
constexpr std::uint32_t PAD_COL_START = 10;
constexpr std::uint32_t PAD_COL_END   = TILE_COLS;

constexpr std::uint32_t FACE_COLS     = 16;
constexpr std::uint32_t FACE_ELEM_CNT = 16 * 16;

static_assert(PAD_COL_START < PAD_COL_END && PAD_COL_END <= TILE_COLS, "pad section must be a non-empty range of tile columns");

constexpr std::uint32_t clamp_to_face(const std::uint32_t col, const std::uint32_t face)
{
    const std::uint32_t face_col0 = face * FACE_COLS;
    return col <= face_col0 ? 0 : (col - face_col0 >= FACE_COLS ? FACE_COLS : col - face_col0);
}

// Face-local column range [start, end) of the pad section inside row 0 of face `face`.
constexpr std::uint32_t seg_start(const std::uint32_t face)
{
    return clamp_to_face(PAD_COL_START, face);
}

constexpr std::uint32_t seg_end(const std::uint32_t face)
{
    return clamp_to_face(PAD_COL_END, face);
}

constexpr std::uint32_t seg_len(const std::uint32_t face)
{
    return seg_end(face) - seg_start(face);
}

constexpr bool face_padded(const std::uint32_t face)
{
    return seg_len(face) != 0;
}

// L1 element offset (16-bit elements) of the first pad element in face `face`.
constexpr std::uint32_t seg_elem_offset(const std::uint32_t face)
{
    return face * FACE_ELEM_CNT + seg_start(face);
}

constexpr std::uint32_t NUM_PADDED_FACES = static_cast<std::uint32_t>(face_padded(0)) + static_cast<std::uint32_t>(face_padded(1));
