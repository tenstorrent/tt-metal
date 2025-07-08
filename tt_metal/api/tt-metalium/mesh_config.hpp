// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/fabric_types.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace tt::tt_metal::distributed {

using chip_id_t = int;

// Configuration for distributed mesh operations across multiple hosts
// Each host manages a subset of the global mesh, defined by local_shape_ and local_offset_
struct DistributedMeshConfig {
    MeshShape global_shape_;    // Total shape of the distributed mesh across all hosts
    MeshShape local_shape_;     // Shape of the mesh portion managed by this host
    MeshCoordinate local_offset_; // Offset of this host's portion within the global mesh

    DistributedMeshConfig(
        const MeshShape& global_shape,
        const MeshShape& local_shape,
        const MeshCoordinate& local_offset) :
        global_shape_(global_shape),
        local_shape_(local_shape),
        local_offset_(local_offset) {
        validate_config();
    }

    // Check if a global coordinate is managed by this host
    bool is_local_coordinate(const MeshCoordinate& global_coord) const {
        // Check if the coordinate falls within this host's local mesh bounds
        for (size_t dim = 0; dim < global_coord.dims(); ++dim) {
            if (global_coord[dim] < local_offset_[dim] || 
                global_coord[dim] >= local_offset_[dim] + local_shape_[dim]) {
                return false;
            }
        }
        return true;
    }

private:
    void validate_config() const {
        // Validate that local mesh fits within global mesh
        TT_FATAL(local_offset_.dims() == global_shape_.dims() && local_offset_.dims() == local_shape_.dims(),
                 "Dimension mismatch between global shape, local shape, and offset");
        for (size_t dim = 0; dim < local_offset_.dims(); ++dim) {
            TT_FATAL(local_offset_[dim] + local_shape_[dim] <= global_shape_[dim],
                     "Local mesh extends beyond global mesh boundaries at dimension {}", dim);
        }
    }
};

// Specifies the configuration of a MeshDevice.
class MeshDeviceConfig {
public:
    // Constructs a MeshDeviceConfig.
    // `offset` is the optional parameter that specifies the offset of the mesh device within the connected system mesh.
    // `physical_device_ids` is the optional parameter that allows to override physical device IDs used to create the
    // mesh device.
    MeshDeviceConfig(
        const MeshShape& mesh_shape,
        const std::optional<MeshCoordinate>& offset = std::nullopt,
        const std::vector<chip_id_t>& physical_device_ids = {}) :
        mesh_shape_(mesh_shape), offset_(offset), physical_device_ids_(physical_device_ids) {}

    const MeshShape& mesh_shape() const { return mesh_shape_; }
    const std::optional<MeshCoordinate>& offset() const { return offset_; }
    const std::vector<chip_id_t>& physical_device_ids() const { return physical_device_ids_; }

private:
    MeshShape mesh_shape_;
    std::optional<MeshCoordinate> offset_;
    std::vector<chip_id_t> physical_device_ids_;
};

}  // namespace tt::tt_metal::distributed
