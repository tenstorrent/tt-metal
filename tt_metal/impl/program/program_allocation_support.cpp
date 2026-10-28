// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "program_allocation_support.hpp"

#include <algorithm>

#include <tt_align.hpp>
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
        } else {
            // TODO(#62451): The following assertion assumes a bump-only allocator. A DFB recorded into an intersecting
            // CoreRange with uneven persistent L1 can sit below that range's base. Commented out for now.
            // TT_ASSERT(
            //     base_address <= region.l1_regions.front().first,
            //     "Persistent L1 on core range {} grew to {}, above its program-local allocations starting at {}",
            //     region.core_range.str(),
            //     base_address,
            //     region.l1_regions.front().first);
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
    for (const CoreRange& core_range : cores.ranges()) {
        for (auto& region : regions) {
            if (region.core_range.intersects(core_range)) {
                // If any allocation happened on the region
                const bool region_has_been_allocated =
                    !region.l1_regions.empty() and region.l1_regions.back().second > region.l1_regions.front().first;

                if (region.core_range != core_range and region_has_been_allocated and
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

DeviceAddr allocate_scratchpad(
    EphemeralAllocationMetadata& metadata, const CoreRangeSet& cores, uint64_t size, uint32_t alignment) {
    if (cores.empty()) {
        return metadata.empty_cores_high_water_mark;
    }

    auto& regions = metadata.regions;
    for (const CoreRange& core_range : cores.ranges()) {
        TT_FATAL(
            std::any_of(
                regions.begin(),
                regions.end(),
                [&core_range](const auto& region) { return region.core_range == core_range; }),
            "Internal error: core range {} was not registered before allocation",
            core_range.str());
    }

    // Take the max region end over EVERY region that intersects `cores` (not just exact-range matches), so the
    // allocation cannot overlap one on an overlapping-but-different core range. Mark each such region exactly once.
    std::vector<EphemeralAllocationMetadata::LockstepAllocationRegion*> touched;
    for (auto& region : regions) {
        if (cores.intersects(region.core_range)) {
            touched.push_back(&region);
        }
    }
    uint64_t addr = 0;
    for (const auto* region : touched) {
        TT_FATAL(
            !region->l1_regions.empty(),
            "Base address of core range {} is unknown. Call prepare_ephemeral_allocation() before allocating.",
            region->core_range.str());
        addr = std::max(addr, region->l1_regions.back().second);
    }
    addr = align(addr, alignment);
    for (auto* region : touched) {
        region->mark_address(addr, size);
    }
    return addr;
}

void invalidate_ephemeral_allocation_metadata(EphemeralAllocationMetadata& metadata) {
    for (auto& region : metadata.regions) {
        region.reset_available_addresses();
    }
}

}  // namespace tt::tt_metal::detail
