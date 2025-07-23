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

    // Returns the physical device IDs mapped to a MeshDevice
    DistributedMeshContainer<int> get_mapped_physical_device_ids(
        const MeshShape& shape, const std::optional<MeshCoordinate>& offset = std::nullopt) const;
};

}  // namespace tt::tt_metal::distributed
