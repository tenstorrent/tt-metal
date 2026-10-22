// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "program_allocation_support.hpp"

#include <algorithm>

#include <tt_align.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::detail {

namespace {

// Returns address for next allocation
// Allocations are placed sequentially on a core so the next available address gets appended to the last L1 region
uint64_t get_region_end(const L1RegionAllocator& allocator) {
    return allocator.l1_regions.empty() ? 0 : allocator.l1_regions.back().second;
}

}  // namespace

void L1RegionAllocator::mark_address(uint64_t address, uint64_t size) {
    if (this->l1_regions.empty()) {
        this->l1_regions.emplace_back(this->base_address, this->base_address);
    }
    auto& last_region = this->l1_regions.back();
    if (address < last_region.second) {
        TT_THROW(
            "Local buffer address {} has to append to last L1 region [{}, {}) or be at a higher address",
            address,
            last_region.first,
            last_region.second);
    }
    if (address == last_region.second) {
        last_region.second += size;
    } else {
        this->l1_regions.emplace_back(address, address + size);
    }
}

void register_ephemeral_allocation(std::vector<L1RegionAllocator>& allocators, const CoreRangeSet& cores) {
    for (const CoreRange& core_range : cores.ranges()) {
        // There is one L1RegionAllocator per unique core range, create one if it does not already exist for
        // current core range
        auto val =
            std::find_if(allocators.begin(), allocators.end(), [&core_range](const L1RegionAllocator& allocator) {
                return allocator.core_range == core_range;
            });
        if (val == allocators.end()) {
            allocators.emplace_back(core_range);
        }
    }
}

void prepare_ephemeral_allocation(
    std::vector<L1RegionAllocator>& allocators,
    PersistentL1Arena& persistent_l1,
    std::unordered_map<CoreCoord, PersistentL1Arena::Seal>& seals) {
    for (L1RegionAllocator& allocator : allocators) {
        const CoreRangeSet cores(allocator.core_range);
        for (const CoreCoord& core : corerange_to_cores(cores)) {
            if (seals.contains(core)) {
                continue;
            }
            seals.emplace(core, persistent_l1.seal(CoreRangeSet(CoreRange(core))));
        }
        allocator.base_address = persistent_l1.high_water_mark(cores);
    }
}

DeviceAddr allocate_ephemeral_memory(
    std::vector<L1RegionAllocator>& allocators, const CoreRangeSet& cores, uint64_t size, uint32_t alignment) {
    uint64_t computed_addr = 0;
    for (const CoreRange& core_range : cores.ranges()) {
        // Need the max available address across all cores the allocation is placed on
        auto it = std::find_if(allocators.begin(), allocators.end(), [&core_range](const L1RegionAllocator& allocator) {
            return allocator.core_range == core_range;
        });
        TT_FATAL(
            it != allocators.end(),
            "Internal error: core range {} was not registered before allocation",
            core_range.str());
        computed_addr = std::max({computed_addr, it->base_address, get_region_end(*it)});
    }
    computed_addr = align(computed_addr, alignment);
    for (const CoreRange& core_range : cores.ranges()) {
        for (L1RegionAllocator& allocator : allocators) {
            if (allocator.core_range.intersects(core_range)) {
                if (allocator.core_range != core_range and computed_addr < get_region_end(allocator)) {
                    // Intersecting core range has already been marked to have allocation at this address. This
                    // could have been marked by an allocation on a core range disjoint from current `core_range`
                    // but also intersecting `allocator.core_range`
                    continue;
                }
                allocator.mark_address(computed_addr, size);
            }
        }
    }
    return computed_addr;
}

}  // namespace tt::tt_metal::detail
