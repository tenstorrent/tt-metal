// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/mesh_coord.hpp>

namespace tt::tt_metal {

// Manages coordinate translation and validation for distributed meshes, consisting of contiguous local meshes.
class DistributedCoordinateTranslator {
public:
    DistributedCoordinateTranslator(
        const distributed::MeshShape& global_shape,
        const distributed::MeshShape& local_shape,
        const distributed::MeshCoordinate& local_offset);

    // Query methods
    const distributed::MeshShape& global_shape() const noexcept { return global_shape_; }
    const distributed::MeshShape& local_shape() const noexcept { return local_shape_; }
    const distributed::MeshCoordinate& local_offset() const noexcept { return local_offset_; }

    // Returns true if the coordinate is local; throws if the coordinate is out of bounds or invalid.
    bool is_local(const distributed::MeshCoordinate& global_coord) const;

    // Returns the global coordinate of the local coordinate; throws if the coordinate is out of bounds or invalid.
    distributed::MeshCoordinate local_to_global(const distributed::MeshCoordinate& local_coord) const;

private:
    distributed::MeshShape global_shape_;
    distributed::MeshShape local_shape_;
    distributed::MeshCoordinate local_offset_;
};

}  // namespace tt::tt_metal
