// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/indestructible.hpp>
#include <memory>
#include <optional>
#include <vector>

#include <tt-metalium/mesh_coord.hpp>

namespace tt {
namespace stl {
template <typename T>
class Indestructible;
}  // namespace stl
}  // namespace tt

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

    // Returns the physical mesh ID for a given logical coordinate
    uint32_t get_physical_mesh_id(const MeshCoordinate& coord) const;

    // Returns the global device coordinate for a given physical device ID
    MeshCoordinate get_global_device_coordinate(int physical_device_id) const;

    // Returns the physical device IDs mapped to a MeshDevice
    std::vector<int> get_mapped_physical_device_ids(
        const MeshShape& shape, const std::optional<MeshCoordinate>& offset = std::nullopt) const;
};

}  // namespace tt::tt_metal::distributed
