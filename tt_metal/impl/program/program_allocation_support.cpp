// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "program_allocation_support.hpp"

#include <algorithm>

#include <tt-metalium/math.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::detail {

void EphemeralAllocationMetadata::LockstepAllocationRegion::mark_address(uint64_t address, uint64_t size) {
    TT_FATAL(
        !this->l1_regions.empty(),
        "Base address of core range {} is unknown. Call prepare_ephemeral_allocation() before allocating.",
        this->core_range.str());
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

void register_ephemeral_allocation(EphemeralAllocationMetadata& metadata, const CoreRangeSet& cores) {
    auto& regions = metadata.regions;
    for (const CoreRange& core_range : cores.ranges()) {
        // There is one LockstepAllocationRegion per unique core range, create one if it does not already exist for
        // current core range
        auto val = std::find_if(regions.begin(), regions.end(), [&core_range](const auto& region) {
            return region.core_range == core_range;
        });
        if (val == regions.end()) {
            TT_FATAL(
                std::none_of(
                    regions.begin(),
                    regions.end(),
                    [](const auto& region) {
                        return std::any_of(
                            region.l1_regions.begin(), region.l1_regions.end(), [](const auto& interval) {
                                return interval.first < interval.second;
                            });
                    }),
                "Cannot register core range {} while program-local allocations are live. Invalidate them first.",
                core_range.str());
            regions.emplace_back(core_range);
        }
    }
}

void prepare_ephemeral_allocation(EphemeralAllocationMetadata& metadata, PersistentL1Arena& persistent_l1) {
    auto& seals = metadata.seals[&persistent_l1];
    for (auto& region : metadata.regions) {
        const CoreRangeSet cores(region.core_range);
        for (const CoreCoord& core : corerange_to_cores(cores)) {
            if (seals.contains(core)) {
                continue;
            }
            seals.emplace(core, persistent_l1.seal(CoreRangeSet(CoreRange(core))));
        }
        const uint64_t base_address = persistent_l1.high_water_mark(cores);
        if (region.l1_regions.empty()) {
            region.l1_regions.emplace_back(base_address, base_address);
        }
    }

    // Record the water mark for when allocation is requested on an empty core grid.
    metadata.empty_cores_high_water_mark = persistent_l1.high_water_mark(CoreRangeSet());
}

DeviceAddr allocate_dfb(
    EphemeralAllocationMetadata& metadata, const CoreRangeSet& cores, uint64_t size, uint32_t alignment) {
    if (cores.empty()) {
        return metadata.empty_cores_high_water_mark;
    }

    auto& regions = metadata.regions;
    uint64_t computed_addr = 0;
    for (const CoreRange& core_range : cores.ranges()) {
        // Need the max available address across all cores the allocation is placed on
        auto it = std::find_if(regions.begin(), regions.end(), [&core_range](const auto& region) {
            return region.core_range == core_range;
        });
        TT_FATAL(
            it != regions.end(),
            "Internal error: core range {} was not registered before allocation",
            core_range.str());
        TT_FATAL(
            !it->l1_regions.empty(),
            "Base address of core range {} is unknown. Call prepare_ephemeral_allocation() before allocating.",
            core_range.str());
        computed_addr = std::max(computed_addr, it->l1_regions.back().second);
    }
    computed_addr = align(computed_addr, alignment);
    const uint64_t allocation_end = computed_addr + size;
    for (auto& region : regions) {
        if (!cores.intersects(region.core_range)) {
            continue;
        }
        TT_FATAL(
            !region.l1_regions.empty(),
            "Base address of core range {} is unknown. Call prepare_ephemeral_allocation() before allocating.",
            region.core_range.str());
        const uint64_t region_end = region.l1_regions.back().second;
        // An intersecting range can also cover disjoint allocations or higher persistent L1 on other cores.
        // Its waterline must cover this allocation's END even when the start is below that waterline.
        if (allocation_end > region_end) {
            const uint64_t mark_begin = std::max(computed_addr, region_end);
            region.mark_address(mark_begin, allocation_end - mark_begin);
        }
    }
    return computed_addr;
}

DeviceAddr allocate_scratchpad(
    EphemeralAllocationMetadata& metadata, const CoreRangeSet& cores, uint64_t size, uint32_t alignment) {
    return allocate_dfb(metadata, cores, size, alignment);
}

void invalidate_ephemeral_allocation_metadata(EphemeralAllocationMetadata& metadata) {
    for (auto& region : metadata.regions) {
        region.reset_available_addresses();
    }
}

}  // namespace tt::tt_metal::detail
