// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/indestructible.hpp>
#include <memory>
#include <optional>
#include <vector>

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/maybe_remote.hpp>
#include <tt-metalium/routing_table_generator.hpp>

namespace tt::tt_metal::distributed {

// SystemMesh creates a virtualization over the physical devices in the system.
// It creates a logical mesh of devices and manages the mapping between logical and physical device coordinates.
// It serves as a query interface between the logical coordinates to physical device IDs.
class SystemMesh {
private:
    class Impl;  // Forward declaration only

    std::unique_ptr<Impl> pimpl_;
    SystemMesh();

    friend class tt::stl::Indestructible<SystemMesh>;

public:
    static SystemMesh& instance();
    SystemMesh(const SystemMesh&) = delete;
    SystemMesh& operator=(const SystemMesh&) = delete;
    SystemMesh(SystemMesh&&) = delete;
    SystemMesh& operator=(SystemMesh&&) = delete;

    // Returns the shape of the system mesh; this is the global mesh shape in distributed context
    const MeshShape& shape() const;

    // Returns the local shape of the system mesh; this is the local mesh shape in distributed context
    const MeshShape& local_shape() const;

    // Wrapper structure with device IDs, fabric node IDs, and mesh shape ordered in row-major order according to the
    // requested `shape`.
    struct MappedDevices {
        // Device ID is set for host-local devices only.
        std::vector<MaybeRemote<int>> device_ids;

        // Fabric node ID is set for host-local and host-remote devices globally.
        std::vector<tt::tt_fabric::FabricNodeId> fabric_node_ids;

        // Shape of requested mesh if provided, otherwise the system mesh global shape.
        MeshShape mesh_shape;
    };

    // Returns devices that should be mapped to a MeshDevice according to the shape and offset.
    // If `shape` is not provided, the system mesh global shape is used.
    // If `offset` is not provided, an N-dimensional zero-coordinate is used (based on system mesh dims).
    MappedDevices get_mapped_devices(
        const std::optional<MeshShape>& shape, const std::optional<MeshCoordinate>& offset = std::nullopt) const;
};

}  // namespace tt::tt_metal::distributed
