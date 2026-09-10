// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/tile.hpp>

namespace tt::tt_metal {

// Describes the face layout of a tile for a single operand.
//
// A tile is subdivided into equally sized faces. `FaceGeometry` records how many rows each
// face contains (`face_r_dim`) and how many faces make up the operand (`num_faces`). It is
// used to tell the compute engine that an operand's geometry differs from the default
// full-tile layout, for example when data is packed onto compact pages that populate only a
// subset of a tile's faces.
//
// Defaults describe a standard full tile (16-row faces, 4 faces).
struct FaceGeometry {
    uint32_t face_r_dim = constants::FACE_HEIGHT;
    uint32_t num_faces = constants::TILE_HW / constants::FACE_HW;

    bool operator==(const FaceGeometry& other) const {
        return face_r_dim == other.face_r_dim && num_faces == other.num_faces;
    }
};

// The per-CB tile descriptor the JIT bakes into `chlkc_descriptors.h`. Resolved once here
// so the kernel build and the emulator cannot derive it differently.
struct ResolvedTileGeometry {
    Tile tile;  // effective tile: an explicit FaceGeometry substitutes its own
    uint32_t num_faces = constants::TILE_HW / constants::FACE_HW;
    uint32_t face_r_dim = constants::FACE_HEIGHT;
    uint32_t partial_face = 0;
    uint32_t narrow_tile = 0;
};

namespace detail {

inline bool is_supported_tile_shape(uint32_t tile_height, uint32_t tile_width) {
    if (tile_width != constants::FACE_WIDTH && tile_width != constants::TILE_WIDTH) {
        return false;
    }
    return tile_height == 1 || tile_height == 2 || tile_height == 4 || tile_height == 8 ||
           tile_height == constants::FACE_HEIGHT || tile_height == constants::TILE_HEIGHT;
}

inline std::optional<Tile> tile_from_unpack_face_geometry(const FaceGeometry& face_geometry) {
    const uint32_t tile_height =
        face_geometry.face_r_dim * (face_geometry.num_faces > 2 ? constants::TILE_HEIGHT / constants::FACE_HEIGHT : 1);
    const uint32_t tile_width = face_geometry.num_faces == 1 ? constants::FACE_WIDTH : constants::TILE_WIDTH;
    if (!is_supported_tile_shape(tile_height, tile_width)) {
        return std::nullopt;
    }
    return Tile({tile_height, tile_width});
}

}  // namespace detail

// Precedence: an explicit unpack FaceGeometry wins over the CB's Tile, which wins over the
// full-tile default. Callers that only need the tile size may read `.tile`.
inline ResolvedTileGeometry resolve_tile_geometry(
    const std::optional<Tile>& tile, const std::optional<FaceGeometry>& unpack_face_geometry) {
    const Tile default_tile;
    const Tile& requested_tile = tile.value_or(default_tile);
    const std::optional<Tile> face_geometry_tile =
        unpack_face_geometry.has_value() ? detail::tile_from_unpack_face_geometry(*unpack_face_geometry) : std::nullopt;
    const Tile& effective_tile = face_geometry_tile.value_or(requested_tile);
    return ResolvedTileGeometry{
        .tile = effective_tile,
        .num_faces =
            unpack_face_geometry.has_value() ? unpack_face_geometry->num_faces : requested_tile.get_num_faces(),
        .face_r_dim =
            unpack_face_geometry.has_value() ? unpack_face_geometry->face_r_dim : requested_tile.get_face_shape()[0],
        .partial_face = effective_tile.get_partial_face(),
        .narrow_tile = effective_tile.get_narrow_tile(),
    };
}

}  // namespace tt::tt_metal

namespace std {

// Hash support for FaceGeometry (needed for the reflection/hashing system).
template <>
struct hash<tt::tt_metal::FaceGeometry> {
    std::size_t operator()(const tt::tt_metal::FaceGeometry& face_geometry) const noexcept;
};

}  // namespace std
