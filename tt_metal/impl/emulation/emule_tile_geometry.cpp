// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "emule_tile_geometry.hpp"

namespace tt::tt_metal::emule {
namespace {

bool is_supported_tile_shape(uint32_t tile_height, uint32_t tile_width) {
    if (tile_width != tt::constants::FACE_WIDTH && tile_width != tt::constants::TILE_WIDTH) {
        return false;
    }
    return tile_height == 1 || tile_height == 2 || tile_height == 4 || tile_height == 8 ||
           tile_height == tt::constants::FACE_HEIGHT || tile_height == tt::constants::TILE_HEIGHT;
}

std::optional<Tile> tile_from_unpack_face_geometry(const FaceGeometry& face_geometry) {
    const uint32_t tile_height =
        face_geometry.face_r_dim *
        (face_geometry.num_faces > 2 ? tt::constants::TILE_HEIGHT / tt::constants::FACE_HEIGHT : 1);
    const uint32_t tile_width = face_geometry.num_faces == 1 ? tt::constants::FACE_WIDTH : tt::constants::TILE_WIDTH;
    if (!is_supported_tile_shape(tile_height, tile_width)) {
        return std::nullopt;
    }
    return Tile({tile_height, tile_width});
}

}  // namespace

ResolvedTileGeometry resolve_tile_geometry(
    const std::optional<Tile>& tile, const std::optional<FaceGeometry>& unpack_face_geometry) {
    const Tile default_tile;
    const Tile& requested_tile = tile.value_or(default_tile);
    const std::optional<Tile> face_geometry_tile =
        unpack_face_geometry.has_value() ? tile_from_unpack_face_geometry(*unpack_face_geometry) : std::nullopt;
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

}  // namespace tt::tt_metal::emule
