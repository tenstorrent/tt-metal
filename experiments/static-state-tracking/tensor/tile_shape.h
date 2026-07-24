// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// sst::tensor::TileShape
// ----------------------------------------------------------------------------
// Tile layer of the Tensor / Tile / Face stack. A Tile is what the CB lays out
// in L1 and what the data-format register file is configured for; it is one or
// more Faces tiled rectangularly.
// ----------------------------------------------------------------------------

#include <cstdint>

#include "face_shape.h"
#include "format_traits.h"

namespace sst::tensor {

// ----------------------------------------------------------------------------
// Layout: the tile's in-L1 datum ordering.
//   - Tiled:    face-major, row-major within each face.
//   - RowMajor: scan-line order across the whole tile (tilize / untilize ends).
// ----------------------------------------------------------------------------
enum class Layout : uint8_t {
    Tiled = 0,
    RowMajor = 1,
};

// ----------------------------------------------------------------------------
// On-L1 footprint of one tile.
// ----------------------------------------------------------------------------
constexpr uint32_t tile_size_bytes_of(DataFormat F, uint32_t tile_hw) {
    return (F == DataFormat::Float16_b) ? tile_hw * 2u : 0u;
}

// ----------------------------------------------------------------------------
// TileShape: compile-time tile-layer tag.
// ----------------------------------------------------------------------------
template <typename FaceShapeT, uint8_t NumFacesR, uint8_t NumFacesC, Layout L = Layout::Tiled>
struct TileShape {
    using face_shape_t = FaceShapeT;

    static constexpr uint8_t num_faces_r = NumFacesR;
    static constexpr uint8_t num_faces_c = NumFacesC;
    static constexpr uint8_t num_faces = NumFacesR * NumFacesC;
    static constexpr Layout layout = L;

    // Format / face geometry are derived from the FaceShape.
    static constexpr DataFormat data_format = FaceShapeT::data_format;
    static constexpr uint8_t face_r_dim = FaceShapeT::face_r_dim;
    static constexpr uint8_t face_c_dim = FaceShapeT::face_c_dim;

    // Tile dimensions (datums).
    static constexpr uint32_t tile_r_dim = uint32_t(FaceShapeT::face_r_dim) * NumFacesR;
    static constexpr uint32_t tile_c_dim = uint32_t(FaceShapeT::face_c_dim) * NumFacesC;
    static constexpr uint32_t tile_hw = tile_r_dim * tile_c_dim;

    // The canonical on-L1 footprint of one tile (mirrors Tile::get_tile_size).
    static constexpr uint32_t tile_size_bytes() { return tile_size_bytes_of(data_format, tile_hw); }

    static constexpr uint32_t tile_bytes() { return tile_size_bytes(); }
    static constexpr uint32_t tile_words() { return tile_size_bytes() / 16u; }
};

// ----------------------------------------------------------------------------
// The one tile the prototype uses: 32×32 Float16_b (2×2 faces of 16×16).
// ----------------------------------------------------------------------------
using Face16x16_Float16_b = FaceShape<16, 16, DataFormat::Float16_b>;
using Tile32x32_Float16_b = TileShape<Face16x16_Float16_b, 2, 2>;

}  // namespace sst::tensor
