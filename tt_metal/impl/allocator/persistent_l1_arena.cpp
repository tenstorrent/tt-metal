// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/allocator/persistent_l1_arena.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

#include <tt-metalium/tt_align.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_metal {

PersistentL1Arena::PersistentL1Arena(DeviceAddr base, DeviceAddr limit, const CoreRangeSet& worker_grid) :
    base_(base), limit_(limit) {
    TT_FATAL(base_ <= limit_, "Persistent L1 arena has invalid bounds [{}, {})", base_, limit_);
    if (!worker_grid.empty()) {
        const CoreRange bbox = worker_grid.bounding_box();
        grid_width_ = bbox.end_coord.x + 1;
        grid_height_ = bbox.end_coord.y + 1;
        TT_FATAL(grid_width_ > 0 && grid_height_ > 0, "Persistent L1 seal grid has invalid size");
        seal_refcounts_.assign(static_cast<size_t>(grid_width_) * grid_height_, 0);
    }
}

size_t PersistentL1Arena::seal_index(const CoreCoord& core) const {
    TT_FATAL(
        core.x < grid_width_ && core.y < grid_height_,
        "Persistent L1 core {} is outside the worker grid {}x{}",
        core.str(),
        grid_width_,
        grid_height_);
    return static_cast<size_t>(core.y) * grid_width_ + core.x;
}

PersistentL1Arena::Allocation PersistentL1Arena::allocate(
    const CoreRangeSet& cores, DeviceAddr size, DeviceAddr alignment) {
    std::lock_guard lock(mutex_);
    TT_FATAL(!cores.empty(), "Persistent L1 allocation requires at least one core");
    TT_FATAL(size > 0, "Persistent L1 allocation size must be non-zero");
    TT_FATAL(alignment > 0, "Persistent L1 allocation alignment must be non-zero");

    const auto core_list = corerange_to_cores(cores);
    for (const CoreCoord& core : core_list) {
        TT_FATAL(
            seal_refcounts_[seal_index(core)] == 0,
            "Cannot allocate persistent L1 on core {} after program-local L1 placement",
            core.str());
    }

    DeviceAddr candidate = tt::align(base_, alignment);
    while (true) {
        TT_FATAL(
            candidate <= limit_ && size <= limit_ - candidate,
            "Persistent L1 arena is out of memory: cannot place {} bytes aligned to {} on cores {} in [{}, {})",
            size,
            alignment,
            cores.str(),
            base_,
            limit_);

        DeviceAddr next_candidate = candidate;
        for (const CoreCoord& core : core_list) {
            auto regions_it = regions_by_core_.find(core);
            if (regions_it == regions_by_core_.end()) {
                continue;
            }
            for (const Region& region : regions_it->second) {
                if (candidate + size <= region.begin) {
                    break;
                }
                if (candidate < region.end && candidate + size > region.begin) {
                    next_candidate = std::max(next_candidate, tt::align(region.end, alignment));
                    break;
                }
            }
        }
        if (next_candidate != candidate) {
            candidate = next_candidate;
            continue;
        }
        break;
    }

    const uint64_t allocation_id = next_allocation_id_++;
    TT_FATAL(allocation_id != 0, "Persistent L1 allocation id overflow");
    const Region region{candidate, candidate + size, allocation_id};
    for (const CoreCoord& core : core_list) {
        auto& regions = regions_by_core_[core];
        auto pos = std::lower_bound(
            regions.begin(), regions.end(), region.begin, [](const Region& existing, DeviceAddr address) {
                return existing.begin < address;
            });
        regions.insert(pos, region);
    }
    allocations_.emplace(allocation_id, AllocationRecord{cores, candidate, size});
    return Allocation{allocation_id, candidate, size};
}

void PersistentL1Arena::deallocate(uint64_t allocation_id) {
    if (allocation_id == 0) {
        return;
    }
    std::lock_guard lock(mutex_);
    auto allocation_it = allocations_.find(allocation_id);
    TT_FATAL(allocation_it != allocations_.end(), "Unknown persistent L1 allocation id {}", allocation_id);
    for (const CoreCoord& core : corerange_to_cores(allocation_it->second.cores)) {
        auto regions_it = regions_by_core_.find(core);
        TT_FATAL(regions_it != regions_by_core_.end(), "Missing persistent L1 regions for core {}", core.str());
        auto& regions = regions_it->second;
        auto region_it = std::find_if(regions.begin(), regions.end(), [allocation_id](const Region& region) {
            return region.allocation_id == allocation_id;
        });
        TT_FATAL(
            region_it != regions.end(), "Missing persistent L1 allocation {} on core {}", allocation_id, core.str());
        regions.erase(region_it);
        if (regions.empty()) {
            regions_by_core_.erase(regions_it);
        }
    }
    allocations_.erase(allocation_it);
}

DeviceAddr PersistentL1Arena::high_water_mark(const CoreRangeSet& cores) const {
    std::lock_guard lock(mutex_);
    DeviceAddr high_water_mark = base_;
    for (const CoreCoord& core : corerange_to_cores(cores)) {
        auto regions_it = regions_by_core_.find(core);
        if (regions_it != regions_by_core_.end() && !regions_it->second.empty()) {
            high_water_mark = std::max(high_water_mark, regions_it->second.back().end);
        }
    }
    return high_water_mark;
}

std::vector<std::pair<DeviceAddr, DeviceAddr>> PersistentL1Arena::occupied_ranges() const {
    std::lock_guard lock(mutex_);
    std::vector<std::pair<DeviceAddr, DeviceAddr>> ranges;
    for (const auto& [core, regions] : regions_by_core_) {
        (void)core;
        for (const Region& region : regions) {
            ranges.emplace_back(region.begin, region.end);
        }
    }
    std::sort(ranges.begin(), ranges.end());
    ranges.erase(std::unique(ranges.begin(), ranges.end()), ranges.end());
    return ranges;
}

PersistentL1Arena::Seal PersistentL1Arena::seal(const CoreRangeSet& cores) {
    increment_seals(cores);
    return Seal(this, cores);
}

void PersistentL1Arena::increment_seals(const CoreRangeSet& cores) {
    std::lock_guard lock(mutex_);
    for (const CoreCoord& core : corerange_to_cores(cores)) {
        uint32_t& refcount = seal_refcounts_[seal_index(core)];
        TT_FATAL(
            refcount < std::numeric_limits<uint32_t>::max(), "Persistent L1 seal refcount overflow on {}", core.str());
        ++refcount;
    }
}

void PersistentL1Arena::decrement_seals(const CoreRangeSet& cores) {
    std::lock_guard lock(mutex_);
    for (const CoreCoord& core : corerange_to_cores(cores)) {
        uint32_t& refcount = seal_refcounts_[seal_index(core)];
        TT_FATAL(refcount > 0, "Persistent L1 core {} is not sealed", core.str());
        --refcount;
    }
}

PersistentL1Arena::Seal::Seal(PersistentL1Arena* arena, CoreRangeSet cores) : arena_(arena), cores_(std::move(cores)) {}

PersistentL1Arena::Seal::Seal(Seal&& other) noexcept :
    arena_(std::exchange(other.arena_, nullptr)), cores_(std::move(other.cores_)) {}

PersistentL1Arena::Seal& PersistentL1Arena::Seal::operator=(Seal&& other) noexcept {
    if (this != &other) {
        std::swap(arena_, other.arena_);
        std::swap(cores_, other.cores_);
    }
    return *this;
}

PersistentL1Arena::Seal::~Seal() {
    if (arena_ != nullptr) {
        arena_->decrement_seals(cores_);
    }
}

}  // namespace tt::tt_metal
