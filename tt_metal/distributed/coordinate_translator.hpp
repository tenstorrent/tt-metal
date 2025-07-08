// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/shape2d.hpp>

namespace tt::tt_metal::distributed {

// Utility class for translating between global and local coordinate systems
// in a distributed mesh environment where each host manages a portion of the global mesh.
class CoordinateTranslator {
public:
    CoordinateTranslator(const MeshShape& local_shape, const MeshCoordinate& local_offset);

    // Translate global coordinate to local coordinate
    // Returns std::nullopt if the global coordinate is not within the local mesh bounds
    std::optional<MeshCoordinate> global_to_local(const MeshCoordinate& global_coord) const;

    // Translate local coordinate to global coordinate
    MeshCoordinate local_to_global(const MeshCoordinate& local_coord) const;

    // Check if a global coordinate is accessible locally
    bool is_local_coordinate(const MeshCoordinate& global_coord) const;

    // Translate global coordinate to local, or fatal error if out of bounds
    MeshCoordinate translate_or_fatal(const MeshCoordinate& global_coord) const;

    const MeshShape& local_shape() const noexcept;
    const MeshCoordinate& local_offset() const noexcept;

private:
    MeshShape local_shape_;
    MeshCoordinate local_offset_;
};

}  // namespace tt::tt_metal::distributed