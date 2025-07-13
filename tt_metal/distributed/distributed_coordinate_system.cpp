// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/distributed_coordinate_system.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/fabric_types.hpp>
#include <tt-logger/tt-logger.hpp>

namespace tt::tt_metal::distributed {

DistributedCoordinateSystem::DistributedCoordinateSystem(
    const MeshShape& global_shape,
    const MeshShape& local_shape,
    const MeshCoordinate& local_offset)
    : global_shape_(global_shape), local_shape_(local_shape), local_offset_(local_offset) {
    validate_config();
}

void DistributedCoordinateSystem::validate_config() const {
    // Validate dimensions match
    TT_FATAL(local_offset_.dims() == global_shape_.dims() && local_offset_.dims() == local_shape_.dims(),
             "Dimension mismatch between global shape, local shape, and offset");

    // Validate that local mesh fits within global mesh
    for (size_t dim = 0; dim < local_offset_.dims(); ++dim) {
        TT_FATAL(local_offset_[dim] + local_shape_[dim] <= global_shape_[dim],
                 "Local mesh extends beyond global mesh boundaries at dimension {}", dim);
    }
}

bool DistributedCoordinateSystem::is_local(const MeshCoordinate& global_coord) const {
    // Check if the coordinate falls within this host's local mesh bounds
    for (size_t dim = 0; dim < global_coord.dims(); ++dim) {
        if (global_coord[dim] < local_offset_[dim] ||
            global_coord[dim] >= local_offset_[dim] + local_shape_[dim]) {
            return false;
        }
    }
    return true;
}

std::optional<MeshCoordinate> DistributedCoordinateSystem::global_to_local(const MeshCoordinate& global_coord) const {
    tt::stl::SmallVector<uint32_t> local_coord(global_coord.dims());
    for (size_t dim = 0; dim < global_coord.dims(); ++dim) {
        if (global_coord[dim] < local_offset_[dim] ||
            global_coord[dim] >= local_offset_[dim] + local_shape_[dim]) {
            return std::nullopt;
        }
        local_coord[dim] = global_coord[dim] - local_offset_[dim];
    }
    return MeshCoordinate(local_coord);
}

MeshCoordinate DistributedCoordinateSystem::local_to_global(const MeshCoordinate& local_coord) const {
    // Validate input dimensions
    TT_FATAL(local_coord.dims() == local_shape_.dims(),
             "Dimension mismatch: local_coord has {} dimensions, expected {}",
             local_coord.dims(), local_shape_.dims());

    auto global_coord = local_coord;
    for (size_t dim = 0; dim < local_coord.dims(); ++dim) {
        // Validate local coordinate is within bounds
        TT_FATAL(local_coord[dim] < local_shape_[dim],
                 "Local coordinate[{}]={} exceeds local shape[{}]={}",
                 dim, local_coord[dim], dim, local_shape_[dim]);

        global_coord[dim] += local_offset_[dim];
    }
    return MeshCoordinate(global_coord);
}

// Static factory method to create from control plane
DistributedCoordinateSystem DistributedCoordinateSystem::from_control_plane() {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    const auto local_mesh_id = control_plane.get_local_mesh_id_bindings()[0];
    auto local_offset = control_plane.get_local_mesh_offset();

    auto global_shape = control_plane.get_physical_mesh_shape(local_mesh_id, tt::tt_fabric::MeshScope::GLOBAL);
    auto local_shape = control_plane.get_physical_mesh_shape(local_mesh_id, tt::tt_fabric::MeshScope::LOCAL);

    log_debug(LogDistributed,
              "[DistributedCoordinateSystem] Creating from control plane - Global shape: {}, Local shape: {}, Local offset: {}",
              global_shape, local_shape, local_offset);

    return DistributedCoordinateSystem(global_shape, local_shape, local_offset);
}

}  // namespace tt::tt_metal::distributed
