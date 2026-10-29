// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <internal/mesh_device.hpp>
#include <tt-metalium/mesh_device.hpp>
#include "tt_metal/distributed/mesh_device_impl.hpp"

namespace tt::tt_metal::internal {

IDevice* get_device(const distributed::MeshDevice& mesh_device, const distributed::MeshCoordinate& coord) {
    return mesh_device.impl().get_device(coord);
}

IDevice* get_device(const distributed::MeshDevice& mesh_device, ChipId physical_device_id) {
    return mesh_device.impl().get_device(physical_device_id);
}

}  // namespace tt::tt_metal::internal
