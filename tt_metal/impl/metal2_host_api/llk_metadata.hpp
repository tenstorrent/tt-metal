// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/face_geometry.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace tt::tt_metal {

// Host-side metadata needed to specialize an LLK operand.
// Meant to mirror LLKMetadata on kernel side.
struct LLKMetadata {
    DataFormat format;
    Tile tile;
    FaceGeometry face_geometry;
};

inline FaceGeometry FaceGeometryFromTile(const Tile& tile) {
    return FaceGeometry{.face_r_dim = tile.get_face_shape()[0], .num_faces = tile.get_num_faces()};
}

// The tile the LLK sees for an operand: a face geometry override that describes a representable tile
// replaces the operand's own tile, otherwise the operand's tile stands. Only an override may do this --
// a geometry derived from the tile carries no new information, and (face_r_dim=16, num_faces=2) cannot
// say whether it means a wide 16x32 or a narrow 32x16.
//
// Must agree with tile_from_unpack_face_geometry / set_cb_data_fmt_tile_and_face_geometry in
// jit_build/jit_build_options.cpp, which resolve the same thing for the CB-indexed descriptor arrays.
inline Tile EffectiveLlkTile(const std::optional<Tile>& tile, const std::optional<FaceGeometry>& face_geometry) {
    const Tile requested = tile.value_or(Tile{});
    if (!face_geometry.has_value()) {
        return requested;
    }
    const uint32_t tile_height = face_geometry->face_r_dim *
                                 (face_geometry->num_faces > 2 ? constants::TILE_HEIGHT / constants::FACE_HEIGHT : 1);
    const uint32_t tile_width = face_geometry->num_faces == 1 ? constants::FACE_WIDTH : constants::TILE_WIDTH;
    const bool supported = tile_height == 1 || tile_height == 2 || tile_height == 4 || tile_height == 8 ||
                           tile_height == constants::FACE_HEIGHT || tile_height == constants::TILE_HEIGHT;
    return supported ? Tile({tile_height, tile_width}) : requested;
}

}  // namespace tt::tt_metal
