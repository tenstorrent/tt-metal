// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <vector>

#include "mesh_coord.hpp"
#include <tt_stl/indestructible.hpp>

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

    // Returns the shape of the system mesh
    const MeshShape& get_shape() const;

    // Returns the physical device ID for a given logical coordinate
    int get_physical_device_id(const MeshCoordinate& coord) const;

    // Returns the physical device IDs mapped to a MeshDevice
    std::vector<int> get_mapped_physical_device_ids(
        const MeshShape& shape, const std::optional<MeshCoordinate>& offset = std::nullopt) const;
};

}  // namespace tt::tt_metal::distributed
