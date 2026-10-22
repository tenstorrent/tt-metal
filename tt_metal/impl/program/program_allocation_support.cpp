// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "program_allocation_support.hpp"

#include <algorithm>

#include <tt_align.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::detail {

void EphemeralAllocationMetadata::LockstepAllocationRegion::mark_address(uint64_t address, uint64_t size) {
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

void register_ephemeral_allocation(EphemeralAllocationMetadata& metadata, const CoreRangeSet& cores) {
    auto& regions = metadata.regions;
    for (const CoreRange& core_range : cores.ranges()) {
        // There is one LockstepAllocationRegion per unique core range, create one if it does not already exist for
        // current core range
        auto val = std::find_if(regions.begin(), regions.end(), [&core_range](const auto& region) {
            return region.core_range == core_range;
        });
        if (val == regions.end()) {
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
        region.base_address = persistent_l1.high_water_mark(cores);
    }
}

DeviceAddr allocate_ephemeral_memory(
    EphemeralAllocationMetadata& metadata, const CoreRangeSet& cores, uint64_t size, uint32_t alignment) {
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
        const uint64_t region_end = it->l1_regions.empty() ? 0 : it->l1_regions.back().second;
        computed_addr = std::max({computed_addr, it->base_address, region_end});
    }
    computed_addr = align(computed_addr, alignment);
    for (const CoreRange& core_range : cores.ranges()) {
        for (auto& region : regions) {
            if (region.core_range.intersects(core_range)) {
                if (region.core_range != core_range and !region.l1_regions.empty() and
                    computed_addr < region.l1_regions.back().second) {
                    // Intersecting core range has already been marked to have allocation at this address. This
                    // could have been marked by an allocation on a core range disjoint from current `core_range`
                    // but also intersecting `region.core_range`
                    continue;
                }
                region.mark_address(computed_addr, size);
            }
        }
    }
    return computed_addr;
}

void invalidate_ephemeral_allocation_metadata(EphemeralAllocationMetadata& metadata) {
    for (auto& region : metadata.regions) {
        region.reset_available_addresses();
    }
}

}  // namespace tt::tt_metal::detail
