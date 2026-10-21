// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>

namespace tt::tt_metal::experimental::per_core_allocation {

// Return a movable view; the caller retains SRAM storage and validates backing bounds.
// Placements must cover exactly the configured data cores on each selected local device.
distributed::MeshBuffer create_sram_buffer_view(
    distributed::MeshDevice& mesh, const TensorSpec& spec, const std::vector<SRAMTensorViewAddress>& placements);

namespace detail {
// Check the complete padded write range before constructing a view or initializing a semaphore.
void validate_sram_address(IDevice& device, uint64_t address, uint64_t bytes);
}  // namespace detail

// Per-core address for a core on a specific device. A per-core-allocated MeshBuffer allocates
// each device independently, so the address is only defined once a device is named: there is
// deliberately no (mesh_buffer, core) overload, which could only answer for the reference
// (first-local) device and would silently be wrong on every other one.
DeviceAddr get_per_core_address(
    const distributed::MeshBuffer& mesh_buffer, const distributed::MeshCoordinate& device_coord, const CoreCoord& core);

bool is_per_core_allocation(const distributed::MeshBuffer& mesh_buffer);

// Creates a MeshBuffer that only allocates on a single device within the mesh.
std::shared_ptr<distributed::MeshBuffer> create_on_single_device(
    const distributed::MeshBufferConfig& mesh_buffer_config,
    const distributed::DeviceLocalBufferConfig& device_local_config,
    distributed::MeshDevice* mesh_device,
    const distributed::MeshCoordinate& coord);

}  // namespace tt::tt_metal::experimental::per_core_allocation
