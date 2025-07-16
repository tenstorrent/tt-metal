// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/assert.hpp>

namespace tt::tt_metal {

// TODO: #21096 - Remove these aliases once the distributed namespace is removed.
using MeshShape = tt::tt_metal::distributed::MeshShape;
using MeshCoordinate = tt::tt_metal::distributed::MeshCoordinate;

// Wrapper around a MeshShape, arbitrarily distributed over multiple host machines.
class DistributedMeshShape {
public:
    // Creates a fully local distributed shape.
    explicit DistributedMeshShape(const MeshShape& shape);

    // Creates a distributed shape with a global shape, local shape, and local offset.
    // This is a particular case for contiguous local meshes, which can be represented by a shape and an offset.
    DistributedMeshShape(
        const distributed::MeshShape& global_shape,
        const distributed::MeshShape& local_shape,
        const distributed::MeshCoordinate& local_offset);

    // Helper constructor to create a distributed shape from a DistributedMeshContainer.
    // Whether or not specific coordinates are local is determined by the values in the DistributedMeshContainer.
    // In this case, note the local coordinates might not be contiguous.
    template <typename T>
    explicit DistributedMeshShape(const distributed::MeshContainer<distributed::MaybeRemote<T>>& local_markers) :
        local_markers_(local_markers.shape(), 0) {
        int local_count = 0;
        for (const auto& [coord, value] : local_markers) {
            local_markers_.at(coord) = static_cast<uint8_t>(value.is_local());
            local_count += local_markers_.at(coord);
        }
        fully_local_ = local_count == local_markers.shape().mesh_size();
    }

    // Returns the global `MeshShape`.
    const MeshShape& shape() const;

    // Returns true if the shape is fully local.
    bool fully_local() const;

    // Returns true if the coordinate is local.
    bool is_local(const MeshCoordinate& coord) const;

private:
    // Workaround to avoid specializing `MeshContainer<bool>`.
    distributed::MeshContainer<uint8_t> local_markers_;
    bool fully_local_ = false;
};

}  // namespace tt::tt_metal
