// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>
#include <utility>
#include <vector>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>

namespace tt::tt_metal::experimental::per_core_allocation {

using AddressRanges = std::vector<std::pair<DeviceAddr, DeviceAddr>>;

// Occupied L1 [start, end) ranges on one core of one device: lockstep, this core's per-core
// allocations and persistent L1, sorted and coalesced. Requires AllocatorMode::HYBRID.
AddressRanges get_l1_occupied_ranges(
    const distributed::MeshDevice& mesh_device, const distributed::MeshCoordinate& device_coord, const CoreCoord& core);

// Exact allocator-managed free L1 intervals on one core, including allocator shrink bounds and
// excluding lockstep, per-core and persistent reservations. The L1-small partition is outside
// these allocator bounds; trace storage is in DRAM and does not overlap L1.
AddressRanges get_l1_free_ranges(
    const distributed::MeshDevice& mesh_device, const distributed::MeshCoordinate& device_coord, const CoreCoord& core);

// Same query for every core that owns an L1 bank on the device.
std::unordered_map<CoreCoord, AddressRanges> get_l1_occupied_ranges(
    const distributed::MeshDevice& mesh_device, const distributed::MeshCoordinate& device_coord);

}  // namespace tt::tt_metal::experimental::per_core_allocation
