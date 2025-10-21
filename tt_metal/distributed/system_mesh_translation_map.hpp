// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mesh_coord.hpp>
#include <tt-metalium/fabric_types.hpp>
#include <stdint.h>
#include <tuple>

namespace tt {
namespace tt_metal {
namespace distributed {
template <typename T>
class MeshContainer;
}  // namespace distributed
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal::distributed {

// PhysicalMeshCoordinate is a 2D-coordinate in the physical mesh as defined by the Fabric layer.
// MeshCoordinate[0] is the mesh_id and MeshCoordinate[1] is the physical_device_id.
class PhysicalMeshCoordinate {
public:
    using ChipId = uint32_t;
    using MeshId = tt::tt_fabric::MeshId;
    PhysicalMeshCoordinate() = delete;
    PhysicalMeshCoordinate(MeshId mesh_id, ChipId chip_id) : mesh_id_(mesh_id), chip_id_(chip_id) {}
    MeshId mesh_id() const { return mesh_id_; }
    ChipId chip_id() const { return chip_id_; }

    // Needed for reflect / fmt
    static constexpr auto attribute_names = std::forward_as_tuple("mesh_id", "chip_id");
    auto attribute_values() const { return std::forward_as_tuple(mesh_id_, chip_id_); }

private:
    MeshId mesh_id_{0};
    ChipId chip_id_{0};
};

// Returns a map of all physical mesh coordinates in the system.
const MeshContainer<PhysicalMeshCoordinate>& get_system_mesh_coordinate_translation_map();

}  // namespace tt::tt_metal::distributed
