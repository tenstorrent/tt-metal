// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/assert.hpp>
#include <tt_stl/small_vector.hpp>

namespace tt::tt_metal::distributed {

// Manages coordinate translation and validation for distributed mesh operations
class DistributedCoordinateSystem {
public:
    DistributedCoordinateSystem(
        const MeshShape& global_shape,
        const MeshShape& local_shape,
        const MeshCoordinate& local_offset);

    // Query methods
    const MeshShape& global_shape() const noexcept { return global_shape_; }
    const MeshShape& local_shape() const noexcept { return local_shape_; }
    const MeshCoordinate& local_offset() const noexcept { return local_offset_; }

    // Coordinate validation
    bool is_local(const MeshCoordinate& global_coord) const;

    // Coordinate translation
    std::optional<MeshCoordinate> global_to_local(const MeshCoordinate& global_coord) const;
    MeshCoordinate local_to_global(const MeshCoordinate& local_coord) const;

    // Get a range for iterating over local coordinates
    MeshCoordinateRange local_range() const { return MeshCoordinateRange(local_shape_); }

    // Static factory method to create coordinate system from control plane
    static DistributedCoordinateSystem from_control_plane();

private:
    MeshShape global_shape_;
    MeshShape local_shape_;
    MeshCoordinate local_offset_;

    void validate_config() const;
};

}  // namespace tt::tt_metal::distributed
