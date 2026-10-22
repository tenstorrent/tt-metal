// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/hal_types.hpp"  // DeviceAddr

namespace tt::tt_metal::detail {

struct L1RegionAllocator {
    L1RegionAllocator(const CoreRange& core_range_) : core_range(core_range_) {}

    // Circular buffers are created and allocated at core range granularity
    CoreRange core_range;

    // Holds vector of addresses where circular buffers are allocated [start, end)
    // There are multiple ranges because per core L1 regions are not in lockstep but circular buffers spanning
    // multiple cores must share the same address To enable this, circular buffer address is the maximum address
    // amongst all of its target cores This vector is sorted from lower to higher address spaces
    std::vector<std::pair<uint64_t, uint64_t>> l1_regions;

    // Returns address for next circular buffer
    // Circular buffers are placed sequentially on a core so the next available address gets appended to the
    // last L1 region
    uint64_t get_cb_region_end() const { return this->l1_regions.empty() ? 0 : this->l1_regions.back().second; }

    // If address is the end of the last L1 region, the last region is extended by size bytes,
    //  otherwise address must be higher than existing regions and a new L1 region [address, size) is added
    void mark_address(uint64_t address, uint64_t size, uint64_t base_address);

    // Reset when circular buffer allocation is invalidated
    void reset_available_addresses() { this->l1_regions.clear(); }
};

// Adds an allocator for each range of `cores` that does not have one yet.
// Register every core range before the first allocate_ephemeral_memory(), so that allocations on intersecting ranges
// are recorded on it.
void register_ephemeral_allocation(std::vector<L1RegionAllocator>& allocators, const CoreRangeSet& cores);

// Places `size` bytes at one address on every core of `cores` and returns that address.
// The address is aligned to `alignment`, at or above `reserved_base`, and at or above the end of the allocators
// registered for the ranges of `cores`. The allocation is recorded on every allocator whose range intersects
// `cores`; `allocator_base` gives the base address an allocator's first region starts from.
DeviceAddr allocate_ephemeral_memory(
    std::vector<L1RegionAllocator>& allocators,
    const CoreRangeSet& cores,
    uint64_t size,
    DeviceAddr reserved_base,
    uint32_t alignment,
    const std::function<DeviceAddr(const CoreRange&)>& allocator_base);

}  // namespace tt::tt_metal::detail
