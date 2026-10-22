// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "program_allocation_support.hpp"

#include <algorithm>

#include <tt_align.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::detail {

void L1RegionAllocator::mark_address(uint64_t address, uint64_t size, uint64_t base_address) {
    if (this->l1_regions.empty()) {
        this->l1_regions.emplace_back(base_address, base_address);
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

DeviceAddr allocate_ephemeral_memory(
    std::vector<L1RegionAllocator>& allocators,
    const CoreRangeSet& cores,
    uint64_t size,
    DeviceAddr reserved_base,
    uint32_t alignment,
    const std::function<DeviceAddr(const CoreRange&)>& allocator_base) {
    uint64_t computed_addr = reserved_base;
    for (const CoreRange& core_range : cores.ranges()) {
        // Need the max available address across all cores the allocation is placed on
        for (const L1RegionAllocator& allocator : allocators) {
            if (allocator.core_range == core_range) {
                computed_addr = std::max(computed_addr, allocator.get_cb_region_end());
                break;
            }
        }
    }
    computed_addr = align(computed_addr, alignment);
    for (const CoreRange& core_range : cores.ranges()) {
        for (L1RegionAllocator& allocator : allocators) {
            if (allocator.core_range.intersects(core_range)) {
                if (allocator.core_range != core_range and computed_addr < allocator.get_cb_region_end()) {
                    // Intersecting core range has already been marked to have allocation at this address. This
                    // could have been marked by an allocation on a core range disjoint from current `core_range`
                    // but also intersecting `allocator.core_range`
                    continue;
                }
                allocator.mark_address(computed_addr, size, allocator_base(allocator.core_range));
            }
        }
    }
    return computed_addr;
}

}  // namespace tt::tt_metal::detail
