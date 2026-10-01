// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

#include <tt-metalium/constants.hpp>

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

}  // namespace tt::tt_metal

namespace std {

// Hash support for FaceGeometry (needed for the reflection/hashing system).
template <>
struct hash<tt::tt_metal::FaceGeometry> {
    std::size_t operator()(const tt::tt_metal::FaceGeometry& face_geometry) const noexcept;
};

}  // namespace std
