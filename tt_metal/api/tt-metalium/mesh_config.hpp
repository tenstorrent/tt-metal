// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/mesh_coord.hpp>

namespace tt::tt_metal::distributed {

using chip_id_t = int;

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
