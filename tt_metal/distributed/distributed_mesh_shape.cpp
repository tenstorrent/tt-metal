// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/distributed_mesh_shape.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include "tt_metal/distributed/distributed_coordinate_translator.hpp"

namespace tt::tt_metal {

DistributedMeshShape::DistributedMeshShape(const MeshShape& shape) : local_markers_(shape, 1), fully_local_(true) {}

DistributedMeshShape::DistributedMeshShape(
    const distributed::MeshShape& global_shape,
    const distributed::MeshShape& local_shape,
    const distributed::MeshCoordinate& local_offset) :
    local_markers_(global_shape, 0) {
    DistributedCoordinateTranslator coordinate_translator(global_shape, local_shape, local_offset);

    for (const auto& local_coord : distributed::MeshCoordinateRange(local_shape)) {
        local_markers_.at(coordinate_translator.local_to_global(local_coord)) = 1;
    }

    fully_local_ = global_shape == local_shape && local_offset == MeshCoordinate::zero_coordinate(local_shape.dims());
}

const MeshShape& DistributedMeshShape::shape() const { return local_markers_.shape(); }

bool DistributedMeshShape::fully_local() const { return fully_local_; }

bool DistributedMeshShape::is_local(const MeshCoordinate& coord) const {
    TT_FATAL(
        local_markers_.coord_range().contains(coord),
        "Coordinate {} is out of bounds of the distributed shape {}",
        coord,
        local_markers_.shape());
    return local_markers_.at(coord) == 1;
}

}  // namespace tt::tt_metal
