// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/hal_types.hpp"  // DeviceAddr
#include "impl/allocator/persistent_l1_arena.hpp"

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

    // Lowest address allocations may use on this core range, set by prepare_ephemeral_allocation()
    uint64_t base_address = 0;

    // If address is the end of the last L1 region, the last region is extended by size bytes,
    //  otherwise address must be higher than existing regions and a new L1 region [address, size) is added
    void mark_address(uint64_t address, uint64_t size);

    // Reset when circular buffer allocation is invalidated
    void reset_available_addresses() { this->l1_regions.clear(); }
};

// Adds an allocator for each range of `cores` that does not have one yet.
// Register every core range before the first allocate_ephemeral_memory(), so that allocations on intersecting ranges
// are recorded on it.
void register_ephemeral_allocation(std::vector<L1RegionAllocator>& allocators, const CoreRangeSet& cores);

// Sets each allocator's base address to the top of `persistent_l1` on its core range, and seals those cores so no
// persistent allocation can be placed under the ephemeral memory while the seals in `seals` are held. Cores already in
// `seals` are not sealed again. Call before allocate_ephemeral_memory() on `allocators`.
void prepare_ephemeral_allocation(
    std::vector<L1RegionAllocator>& allocators,
    PersistentL1Arena& persistent_l1,
    std::unordered_map<CoreCoord, PersistentL1Arena::Seal>& seals);

// Places `size` bytes at one address on every core of `cores` and returns that address.
// Every range of `cores` must be registered and have its base address set. The address is aligned to `alignment`, at
// or above the base address and the end of the allocators registered for the ranges of `cores`. The allocation is
// recorded on every allocator whose range intersects `cores`.
DeviceAddr allocate_ephemeral_memory(
    std::vector<L1RegionAllocator>& allocators, const CoreRangeSet& cores, uint64_t size, uint32_t alignment);

}  // namespace tt::tt_metal::detail
