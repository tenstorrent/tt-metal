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

// Half-open physical storage owned (or logically exposed by a view) on one core.
struct L1TensorExtent {
    distributed::MeshCoordinate device_coord;
    CoreCoord core_coord;
    DeviceAddr address;
    DeviceAddr size;
};

// Absolute address on one selected device/core; the caller owns and validates the backing storage.
struct L1TensorViewPlacement {
    distributed::MeshCoordinate device_coord;
    CoreCoord core_coord;
    DeviceAddr address;

    L1TensorViewPlacement(distributed::MeshCoordinate device, CoreCoord core, DeviceAddr address) :
        device_coord(std::move(device)), core_coord(core), address(address) {}
};

// Ordinary lockstep owners cover every worker core, range-lockstep/per-core owners
// cover their allocation grids, and non-owning views expose only their logical shards.
std::vector<L1TensorExtent> get_l1_tensor_owner_extents(
    const distributed::MeshBuffer& buffer, const std::vector<distributed::MeshCoordinate>& coords);

// Creates a non-owning view. The caller must retain backing storage until all uses finish.
// Placements cover every shard of each selected device; this API cannot validate backing bounds.
std::shared_ptr<distributed::MeshBuffer> create_l1_tensor_view(
    distributed::MeshDevice& mesh,
    const distributed::MeshBufferConfig& mesh_config,
    const distributed::DeviceLocalBufferConfig& local_config,
    const std::vector<L1TensorViewPlacement>& placements);

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
