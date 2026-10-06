// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <vector>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace tt::tt_metal::experimental::per_core_allocation {

struct L1PoolExtent {
    distributed::MeshCoordinate device_coord;
    CoreCoord core_coord;
    DeviceAddr address = 0;
    DeviceAddr size = 0;
    bool externally_owned = false;

    L1PoolExtent(distributed::MeshCoordinate device, CoreCoord core, DeviceAddr address, DeviceAddr size) :
        device_coord(std::move(device)), core_coord(core), address(address), size(size) {}
};

struct L1PoolPlacement {
    distributed::MeshCoordinate device_coord;
    CoreCoord core_coord;
    size_t owner_index = 0;
    DeviceAddr offset = 0;

    L1PoolPlacement(distributed::MeshCoordinate device, CoreCoord core, size_t owner, DeviceAddr offset) :
        device_coord(std::move(device)), core_coord(core), owner_index(owner), offset(offset) {}
};

class L1Pool {
public:
    const std::vector<L1PoolExtent>& extents() const { return extents_; }
    const std::shared_ptr<distributed::MeshDevice>& mesh_device() const { return mesh_device_; }

private:
    std::shared_ptr<distributed::MeshDevice> mesh_device_;
    std::vector<L1PoolExtent> extents_;
    std::vector<std::shared_ptr<Buffer>> reserved_owners_;
    std::vector<std::shared_ptr<void>> external_owner_pins_;

    friend std::shared_ptr<L1Pool> reserve_l1_pool(
        distributed::MeshDevice&, const std::vector<L1PoolExtent>&);
    friend size_t adopt_l1_pool_extent(
        const std::shared_ptr<L1Pool>&, const L1PoolExtent&, std::shared_ptr<void>);
    friend std::shared_ptr<distributed::MeshBuffer> create_l1_pool_view(
        const std::shared_ptr<L1Pool>&,
        const distributed::MeshBufferConfig&,
        const distributed::DeviceLocalBufferConfig&,
        const std::vector<L1PoolPlacement>&);
};

std::shared_ptr<L1Pool> reserve_l1_pool(
    distributed::MeshDevice& mesh_device, const std::vector<L1PoolExtent>& extents);

// Append storage owned elsewhere. owner_pin must retain that storage for the pool lifetime.
size_t adopt_l1_pool_extent(
    const std::shared_ptr<L1Pool>& pool, const L1PoolExtent& extent, std::shared_ptr<void> owner_pin);

// Retain the actual root allocation behind a MeshBuffer. For lockstep storage this also
// acquires counted per-device mirror leases, so explicit deallocation of the original tensor
// cannot make its still-live storage available to per-core allocation.
std::shared_ptr<void> retain_l1_pool_owner(
    const distributed::MeshBuffer& mesh_buffer,
    const std::vector<distributed::MeshCoordinate>& device_coords);

// Return the exact L1 ranges exposed by this MeshBuffer. Ordinary lockstep
// owners reserve their address on every worker core; range-lockstep and
// per-core owners cover only their allocated cores. Non-owning pool/view
// buffers expose only their logical shard bytes, never the parent capacity.
std::vector<L1PoolExtent> get_l1_pool_owner_extents(
    const distributed::MeshBuffer& mesh_buffer,
    const std::vector<distributed::MeshCoordinate>& device_coords);

std::shared_ptr<distributed::MeshBuffer> create_l1_pool_view(
    const std::shared_ptr<L1Pool>& pool,
    const distributed::MeshBufferConfig& mesh_buffer_config,
    const distributed::DeviceLocalBufferConfig& device_local_config,
    const std::vector<L1PoolPlacement>& placements);

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
