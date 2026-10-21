// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/allocator/persistent_l1_arena.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_metal {

PersistentL1Arena::PersistentL1Arena(DeviceAddr base, DeviceAddr limit, const CoreRangeSet& worker_grid) :
    liveness_(std::make_shared<PersistentL1Arena*>(this)), base_(base), limit_(limit) {
    TT_FATAL(base_ <= limit_, "Persistent L1 arena has invalid bounds [{}, {})", base_, limit_);
    if (!worker_grid.empty()) {
        const CoreRange bbox = worker_grid.bounding_box();
        grid_width_ = bbox.end_coord.x + 1;
        grid_height_ = bbox.end_coord.y + 1;
        TT_FATAL(grid_width_ > 0 && grid_height_ > 0, "Persistent L1 seal grid has invalid size");
        seals_.resize(static_cast<size_t>(grid_width_) * grid_height_);
    }
}

PersistentL1Arena::~PersistentL1Arena() {
    // Outstanding Seals may still be alive (cached Programs are destroyed after the
    // owning device tears the allocator down). Null the token first so their destructors
    // observe a dead arena even while this one is still unwinding.
    if (liveness_) {
        *liveness_ = nullptr;
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
    // Holders to relocate once the region is placed. Collected under the lock, relocated after
    // it is released: relocating releases the holder's Seals, which takes the lock again.
    std::vector<std::shared_ptr<Relocatable>> holders_to_relocate;
    Allocation allocation;
    {
        std::lock_guard lock(mutex_);
        allocation = allocate_locked(cores, size, alignment, holders_to_relocate);
    }
    for (const auto& holder : holders_to_relocate) {
        holder->relocate_persistent_l1(*this);
    }
    return allocation;
}

PersistentL1Arena::Allocation PersistentL1Arena::allocate_locked(
    const CoreRangeSet& cores,
    DeviceAddr size,
    DeviceAddr alignment,
    std::vector<std::shared_ptr<Relocatable>>& holders_to_relocate) {
    TT_FATAL(!cores.empty(), "Persistent L1 allocation requires at least one core");
    TT_FATAL(size > 0, "Persistent L1 allocation size must be non-zero");
    TT_FATAL(alignment > 0, "Persistent L1 allocation alignment must be non-zero");

    const auto core_list = corerange_to_cores(cores);
    // Check every core before anything moves, so a fixed seal on any core leaves every
    // relocatable holder where it is.
    std::unordered_set<const Relocatable*> sealed_by;
    for (const CoreCoord& core : core_list) {
        const CoreSeals& core_seals = seals_[seal_index(core)];
        TT_FATAL(
            core_seals.num_fixed == 0,
            "Cannot allocate persistent L1 on core {} after program-local L1 placement: a sub-device local L1 slab "
            "is sealed there until its SubDeviceManager is removed",
            core.str());
        sealed_by.insert(core_seals.relocatable.begin(), core_seals.relocatable.end());
    }
    holders_to_relocate.reserve(sealed_by.size());
    for (const Relocatable* key : sealed_by) {
        auto holder_it = holders_.find(key);
        TT_FATAL(holder_it != holders_.end(), "Persistent L1 seal registry has no record for a sealing holder");
        // Expired: the holder is being destroyed and will never launch again; its destructor
        // releases its Seals.
        if (auto holder = holder_it->second.holder.lock()) {
            holders_to_relocate.push_back(std::move(holder));
        }
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

bool PersistentL1Arena::would_relocate(const CoreRangeSet& cores) const {
    std::lock_guard lock(mutex_);
    bool any_relocatable = false;
    for (const CoreCoord& core : corerange_to_cores(cores)) {
        const CoreSeals& core_seals = seals_[seal_index(core)];
        if (core_seals.num_fixed > 0) {
            return false;
        }
        any_relocatable = any_relocatable || !core_seals.relocatable.empty();
    }
    return any_relocatable;
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

std::vector<std::pair<DeviceAddr, DeviceAddr>> PersistentL1Arena::occupied_ranges(const CoreCoord& core) const {
    std::lock_guard lock(mutex_);
    std::vector<std::pair<DeviceAddr, DeviceAddr>> ranges;
    auto regions_it = regions_by_core_.find(core);
    if (regions_it == regions_by_core_.end()) {
        return ranges;
    }
    ranges.reserve(regions_it->second.size());
    for (const Region& region : regions_it->second) {
        ranges.emplace_back(region.begin, region.end);
    }
    return ranges;
}

PersistentL1Arena::Seal PersistentL1Arena::seal(const CoreRangeSet& cores) {
    std::lock_guard lock(mutex_);
    add_seal_locked(cores, nullptr);
    return Seal(liveness_, cores, nullptr);
}

PersistentL1Arena::Seal PersistentL1Arena::seal(const CoreRangeSet& cores, const std::weak_ptr<Relocatable>& holder) {
    const std::shared_ptr<Relocatable> live_holder = holder.lock();
    TT_FATAL(live_holder != nullptr, "A relocatable persistent L1 seal needs a holder owned by a shared_ptr");
    const Relocatable* key = live_holder.get();
    std::lock_guard lock(mutex_);
    const uint64_t num_core_seals = add_seal_locked(cores, key);
    // The record holds a weak reference, so a make_shared holder's storage (and thus this key)
    // cannot be reused until its last Seal is released and the record erased.
    HolderRecord& record = holders_.try_emplace(key, HolderRecord{.holder = holder}).first->second;
    record.num_core_seals += num_core_seals;
    return Seal(liveness_, cores, key);
}

uint64_t PersistentL1Arena::add_seal_locked(const CoreRangeSet& cores, const Relocatable* holder) {
    const auto core_list = corerange_to_cores(cores);
    std::vector<size_t> indices;
    indices.reserve(core_list.size());
    // Resolve every core before changing any, so an off-grid core leaves the registry untouched.
    for (const CoreCoord& core : core_list) {
        indices.push_back(seal_index(core));
        TT_FATAL(
            holder != nullptr || seals_[indices.back()].num_fixed < std::numeric_limits<uint32_t>::max(),
            "Persistent L1 seal refcount overflow on {}",
            core.str());
    }
    for (const size_t index : indices) {
        if (holder == nullptr) {
            ++seals_[index].num_fixed;
        } else {
            seals_[index].relocatable.push_back(holder);
        }
    }
    return indices.size();
}

void PersistentL1Arena::release_seal(const CoreRangeSet& cores, const Relocatable* holder) {
    std::lock_guard lock(mutex_);
    uint64_t num_core_seals = 0;
    for (const CoreCoord& core : corerange_to_cores(cores)) {
        CoreSeals& core_seals = seals_[seal_index(core)];
        ++num_core_seals;
        if (holder == nullptr) {
            TT_FATAL(core_seals.num_fixed > 0, "Persistent L1 core {} has no fixed seal to release", core.str());
            --core_seals.num_fixed;
            continue;
        }
        auto& entries = core_seals.relocatable;
        auto entry_it = std::find(entries.begin(), entries.end(), holder);
        TT_FATAL(entry_it != entries.end(), "Persistent L1 core {} is not sealed by this holder", core.str());
        *entry_it = entries.back();
        entries.pop_back();
    }
    if (holder == nullptr) {
        return;
    }
    auto holder_it = holders_.find(holder);
    TT_FATAL(
        holder_it != holders_.end() && holder_it->second.num_core_seals >= num_core_seals,
        "Persistent L1 seal registry is missing seals for a releasing holder");
    holder_it->second.num_core_seals -= num_core_seals;
    if (holder_it->second.num_core_seals == 0) {
        holders_.erase(holder_it);
    }
}

PersistentL1Arena::Seal::Seal(
    const std::shared_ptr<PersistentL1Arena*>& arena, CoreRangeSet cores, const Relocatable* holder) :
    arena_(arena), cores_(std::move(cores)), holder_(holder) {}

PersistentL1Arena::Seal::Seal(Seal&& other) noexcept :
    arena_(std::move(other.arena_)), cores_(std::move(other.cores_)), holder_(std::exchange(other.holder_, nullptr)) {
    other.arena_.reset();
}

PersistentL1Arena::Seal& PersistentL1Arena::Seal::operator=(Seal&& other) noexcept {
    if (this != &other) {
        std::swap(arena_, other.arena_);
        std::swap(cores_, other.cores_);
        std::swap(holder_, other.holder_);
    }
    return *this;
}

PersistentL1Arena::Seal::~Seal() {
    auto arena = arena_.lock();
    arena_.reset();
    if (!arena || *arena == nullptr) {
        return;
    }
    try {
        (*arena)->release_seal(cores_, holder_);
    } catch (...) {
        log_warning(tt::LogMetal, "Persistent L1 seal release failed on cores {}", cores_.str());
    }
}

}  // namespace tt::tt_metal
