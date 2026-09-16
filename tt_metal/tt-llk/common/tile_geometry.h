// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ckernel
{

// Face grid (rows x columns). Face height is independent of this geometry.
enum class TileGeometry : std::uint8_t
{
    Faces1x1, // One face, including partial-height faces.
    Faces1x2, // Two horizontal faces.
    Faces2x1, // Two vertical faces.
    Faces2x2, // Four faces in row-major order.
};

// Each dimension of the face grid must be 1 or 2.
constexpr TileGeometry get_tile_geometry(const std::uint32_t num_faces_r_dim, const std::uint32_t num_faces_c_dim)
{
    if (num_faces_r_dim == 1)
    {
        return num_faces_c_dim == 1 ? TileGeometry::Faces1x1 : TileGeometry::Faces1x2;
    }
    return num_faces_c_dim == 1 ? TileGeometry::Faces2x1 : TileGeometry::Faces2x2;
}

} // namespace ckernel
