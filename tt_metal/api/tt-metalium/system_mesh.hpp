// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <vector>

#include "mesh_config.hpp"
#include "mesh_coord.hpp"
#include <tt_stl/indestructible.hpp>
namespace tt::tt_metal::distributed {

// SystemMesh creates a virtualization over the physical devices in the system.
// It creates a logical 2D-mesh of devices and manages the mapping between logical and physical device coordinates.
// It serves as a query interface between the logical 2D coordinates to physical device IDs.
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

    // Returns the physical device ID for a given logical row and column index
    chip_id_t get_physical_device_id(const MeshCoordinate& coord) const;

    // Returns the physical device IDs mapped to a MeshDevice
    std::vector<chip_id_t> get_mapped_physical_device_ids(const MeshDeviceConfig& config) const;
    std::vector<chip_id_t> request_available_devices(const MeshDeviceConfig& config) const;
};

}  // namespace tt::tt_metal::distributed
