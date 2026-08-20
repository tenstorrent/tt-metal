// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include <fmt/format.h>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/face_geometry.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_metal {

// The compile-time half of an LLK operand, normalized off a DFB / scratchpad / tensor spec. Baked onto
// the kernel's binding token by genfiles (as device-side `LlkOperandMembers`) and hashed into the kernel
// cache key. `present` is false when the source declared no data format, in which case the remaining
// fields are meaningless and genfiles emits the token's no-metadata constructor.
struct LlkOperandFacts {
    uint8_t hw_format = 0;
    uint8_t face_r_dim = 16;
    uint8_t face_c_dim = 16;
    uint8_t num_faces_r_dim = 2;
    uint8_t num_faces_c_dim = 2;
    bool present = false;
};

struct FaceGridDims {
    uint32_t face_r_dim = constants::FACE_HEIGHT;
    uint32_t face_c_dim = constants::FACE_WIDTH;
    uint32_t num_faces_r_dim = 2;
    uint32_t num_faces_c_dim = 2;
};

// A few host DataFormat enumerators use a value that differs from the HW encoding, to keep host enum
// values unique (e.g. UInt16 owns 9 on host, so host Int16 is 13). Device compilation needs the real HW
// code. Keep in sync with tensix_types.h.
inline uint8_t host_data_format_to_hw(tt::DataFormat format) {
    switch (format) {
        case tt::DataFormat::Int16: return 9;
        case tt::DataFormat::MxFp4_2x_B: return 24;
        case tt::DataFormat::MxInt8: return 2;
        case tt::DataFormat::MxInt4: return 3;
        case tt::DataFormat::MxInt2: return 11;
        default: return static_cast<uint8_t>(format);
    }
}

namespace detail {
inline bool is_supported_tile_shape(uint32_t tile_height, uint32_t tile_width) {
    if (tile_width != constants::FACE_WIDTH && tile_width != constants::TILE_WIDTH) {
        return false;
    }
    return tile_height == 1 || tile_height == 2 || tile_height == 4 || tile_height == 8 ||
           tile_height == constants::FACE_HEIGHT || tile_height == constants::TILE_HEIGHT;
}
}  // namespace detail

// The tile a FaceGeometry override implies, when the override describes a shape Tile can represent.
// nullopt means "keep the requested tile and only shorten its faces".
inline std::optional<Tile> tile_from_unpack_face_geometry(const FaceGeometry& face_geometry) {
    const uint32_t tile_height =
        face_geometry.face_r_dim * (face_geometry.num_faces > 2 ? constants::TILE_HEIGHT / constants::FACE_HEIGHT : 1);
    const uint32_t tile_width = face_geometry.num_faces == 1 ? constants::FACE_WIDTH : constants::TILE_WIDTH;
    if (!detail::is_supported_tile_shape(tile_height, tile_width)) {
        return std::nullopt;
    }
    return Tile({tile_height, tile_width});
}

// Split a flat (face_r_dim, num_faces) pair into the 2D face grid the LLK TensorShape wants, and reject
// combinations that would not fit the tile. `context` names the offending operand in the error message.
inline FaceGridDims compute_face_grid_dims(
    uint32_t tile_r_dim,
    uint32_t tile_c_dim,
    uint32_t face_r_dim,
    uint32_t num_faces,
    std::string_view context = "LLK operand") {
    TT_FATAL(face_r_dim > 0, "{}: face_r_dim must be > 0", context);
    TT_FATAL(num_faces > 0, "{}: num_faces must be > 0", context);
    TT_FATAL(
        tile_c_dim % constants::FACE_WIDTH == 0,
        "{}: tile_c_dim ({}) must be a multiple of FACE_WIDTH ({})",
        context,
        tile_c_dim,
        constants::FACE_WIDTH);
    const uint32_t tile_c_faces = tile_c_dim / constants::FACE_WIDTH;
    TT_FATAL(tile_c_faces > 0, "{}: tile_c_dim ({}) must include at least one face", context, tile_c_dim);
    FaceGridDims grid;
    grid.face_r_dim = face_r_dim;
    grid.face_c_dim = constants::FACE_WIDTH;
    grid.num_faces_c_dim = std::min(tile_c_faces, num_faces);
    TT_FATAL(
        num_faces % grid.num_faces_c_dim == 0,
        "{}: num_faces ({}) must be divisible by num_faces_c_dim ({})",
        context,
        num_faces,
        grid.num_faces_c_dim);
    grid.num_faces_r_dim = num_faces / grid.num_faces_c_dim;
    // Guard against bogus (face_r_dim, num_faces) combos: the logical face grid must fit within the tile
    // rows. e.g. (face_r_dim=9, num_faces=8) on a 32x32 tile would produce num_faces_r_dim=4 -> 36 rows,
    // overflowing the tile and corrupting downstream face addressing.
    TT_FATAL(
        grid.num_faces_r_dim * face_r_dim <= tile_r_dim,
        "{}: face grid (num_faces_r_dim={} * face_r_dim={} = {} rows) exceeds tile_r_dim ({}) "
        "(num_faces={}, num_faces_c_dim={}, tile_c_dim={})",
        context,
        grid.num_faces_r_dim,
        face_r_dim,
        grid.num_faces_r_dim * face_r_dim,
        tile_r_dim,
        num_faces,
        grid.num_faces_c_dim,
        tile_c_dim);
    return grid;
}

inline LlkOperandFacts make_llk_operand_facts(tt::DataFormat host_fmt, const FaceGridDims& grid) {
    return LlkOperandFacts{
        .hw_format = host_data_format_to_hw(host_fmt),
        .face_r_dim = static_cast<uint8_t>(grid.face_r_dim),
        .face_c_dim = static_cast<uint8_t>(grid.face_c_dim),
        .num_faces_r_dim = static_cast<uint8_t>(grid.num_faces_r_dim),
        .num_faces_c_dim = static_cast<uint8_t>(grid.num_faces_c_dim),
        .present = true};
}

inline LlkOperandFacts facts_from_tile(tt::DataFormat host_fmt, const Tile& tile) {
    return make_llk_operand_facts(
        host_fmt,
        compute_face_grid_dims(tile.get_height(), tile.get_width(), tile.get_face_shape()[0], tile.get_num_faces()));
}

// DFB / scratchpad path: an optional storage tile plus an optional face override for operands that occupy
// less than that tile. Tensors never take this path -- their tile is the geometry.
inline LlkOperandFacts facts_from_format_tile_and_face(
    tt::DataFormat host_fmt, const std::optional<Tile>& tile, const std::optional<FaceGeometry>& face) {
    const Tile default_tile;
    const Tile& requested_tile = tile.value_or(default_tile);
    if (!face.has_value()) {
        return facts_from_tile(host_fmt, requested_tile);
    }
    const std::optional<Tile> face_geometry_tile = tile_from_unpack_face_geometry(*face);
    const Tile& effective_tile = face_geometry_tile.value_or(requested_tile);
    return make_llk_operand_facts(
        host_fmt,
        compute_face_grid_dims(
            effective_tile.get_height(), effective_tile.get_width(), face->face_r_dim, face->num_faces));
}

// Designated-initializer text for the device-side `LlkOperandMembers` aggregate (llk_operand_members.h).
// Positional init is deliberately avoided: a field reorder would silently swap geometry. Both the JIT
// filegen and the emulation twin emit through here so the two cannot drift.
inline std::string format_llk_operand_members(const LlkOperandFacts& facts) {
    return fmt::format(
        "{{.format = {}u, .face_r_dim = {}u, .face_c_dim = {}u, .num_faces_r_dim = {}u, .num_faces_c_dim = {}u}}",
        facts.hw_format,
        facts.face_r_dim,
        facts.face_c_dim,
        facts.num_faces_r_dim,
        facts.num_faces_c_dim);
}

}  // namespace tt::tt_metal
