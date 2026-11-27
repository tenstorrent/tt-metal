// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/device_types.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace tt::tt_metal {
class IDevice;
}  // namespace tt::tt_metal

namespace tt::tt_metal::distributed {
class MeshDevice;
}  // namespace tt::tt_metal::distributed

// Internal MeshDevice accessors. Not part of the public API; no stability guarantee.
namespace tt::tt_metal::internal {

// Returns the IDevice at `coord`, or nullptr if `coord` is outside the mesh. Throws if the device is remote.
IDevice* get_device(const distributed::MeshDevice& mesh_device, const distributed::MeshCoordinate& coord);

// Returns the IDevice with physical id `physical_device_id`. Throws if it is not in the mesh or is remote.
IDevice* get_device(const distributed::MeshDevice& mesh_device, ChipId physical_device_id);

}  // namespace tt::tt_metal::internal
