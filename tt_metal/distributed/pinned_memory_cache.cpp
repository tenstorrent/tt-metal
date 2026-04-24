// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "pinned_memory_cache.hpp"

#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <context/metal_context.hpp>
#include "llrt/tt_cluster.hpp"
#include "distributed/mesh_device_view_impl.hpp"

namespace tt::tt_metal::experimental {

using ChipId = int;

PinnedMemoryCache& PinnedMemoryCache::instance() {
    static PinnedMemoryCache cache;
    return cache;
}

std::set<ChipId> PinnedMemoryCache::compute_device_ids(
    distributed::MeshDevice& mesh_device, const distributed::MeshCoordinateRangeSet& coordinate_range_set) {
    const auto& view = mesh_device.get_view();
    std::set<ChipId> device_ids;
    for (const auto& coord : coordinate_range_set.coords()) {
        if (view.contains(coord)) {
            // get_device is deprecated but still functional; we only use it to
            // obtain the chip ID, which is not distribution-sensitive.
            if (auto* device = view.impl().get_device(coord)) {
                device_ids.insert(device->id());
            }
        }
    }
    return device_ids;
}

std::set<ChipId> PinnedMemoryCache::compute_mmio_device_ids(const std::set<ChipId>& device_ids) {
    auto& cluster = MetalContext::instance().get_cluster();
    std::set<ChipId> mmio_ids;
    for (ChipId device_id : device_ids) {
        mmio_ids.insert(cluster.get_associated_mmio_device(device_id));
    }
    return mmio_ids;
}

void PinnedMemoryCache::erase_entry(std::list<CacheEntry>::iterator it) {
    auto [range_begin, range_end] = address_map_.equal_range(it->host_address);
    for (auto map_it = range_begin; map_it != range_end; ++map_it) {
        if (map_it->second == it) {
            address_map_.erase(map_it);
            break;
        }
    }
    lru_entries_.erase(it);
}

bool PinnedMemoryCache::has_conflicting_entry(const void* host_addr, const std::set<ChipId>& target_mmio_ids) const {
    auto [range_begin, range_end] = address_map_.equal_range(host_addr);
    for (auto map_it = range_begin; map_it != range_end; ++map_it) {
        auto list_it = map_it->second;
        if (list_it->pinned_memory.use_count() <= 1) {
            continue;
        }
        for (ChipId mmio_id : list_it->mmio_device_ids) {
            if (target_mmio_ids.contains(mmio_id)) {
                return true;
            }
        }
    }
    return false;
}

std::shared_ptr<PinnedMemory> PinnedMemoryCache::try_pin(
    distributed::MeshDevice& mesh_device,
    const distributed::MeshCoordinateRangeSet& coordinate_range_set,
    HostBuffer& host_buffer,
    bool map_to_noc) {
    // Check whether this hardware/IOMMU configuration supports pinning at all.
    const auto params = GetMemoryPinningParameters(mesh_device);
    if (params.max_pins == 0) {
        return nullptr;
    }
    if (map_to_noc && !params.can_map_to_noc) {
        return nullptr;
    }

    const auto buffer_bytes = host_buffer.view_bytes();
    if (buffer_bytes.empty()) {
        return nullptr;
    }
    const void* host_addr = static_cast<const void*>(buffer_bytes.data());
    const size_t buffer_size = buffer_bytes.size();
    std::set<ChipId> target_device_ids = compute_device_ids(mesh_device, coordinate_range_set);
    if (target_device_ids.empty()) {
        return nullptr;
    }

    std::set<ChipId> target_mmio_ids = compute_mmio_device_ids(target_device_ids);

    std::lock_guard<std::mutex> lock(mutex_);

    // Search all entries for this host address for a usable cache hit.
    auto [range_begin, range_end] = address_map_.equal_range(host_addr);
    for (auto map_it = range_begin; map_it != range_end; ++map_it) {
        auto list_it = map_it->second;
        if (list_it->pinned_memory->get_buffer_size() < buffer_size) {
            continue;
        }
        if (map_to_noc && !list_it->map_to_noc) {
            continue;
        }
        bool covers_all_devices = true;
        for (ChipId device_id : target_device_ids) {
            if (!list_it->device_ids.contains(device_id)) {
                covers_all_devices = false;
                break;
            }
        }
        if (covers_all_devices) {
            lru_entries_.splice(lru_entries_.end(), lru_entries_, list_it);
            return list_it->pinned_memory;
        }
    }

    // No usable entry. Before creating a new pin, check whether any existing
    // entry for this address is still referenced externally on an MMIO device
    // we need. The kernel driver will refuse to pin the same range twice on
    // the same MMIO device, so we must give up if there is a conflict.
    if (has_conflicting_entry(host_addr, target_mmio_ids)) {
        return nullptr;
    }

    // Evict existing entries for this address whose MMIO sets overlap with the
    // target, since they would conflict with the new pin. Entries on disjoint
    // MMIO devices are kept — they are still useful for future lookups.
    {
        auto [rb, re] = address_map_.equal_range(host_addr);
        for (auto map_it = rb; map_it != re;) {
            auto list_it = map_it->second;
            bool mmio_overlap = false;
            for (ChipId mmio_id : list_it->mmio_device_ids) {
                if (target_mmio_ids.contains(mmio_id)) {
                    mmio_overlap = true;
                    break;
                }
            }
            if (mmio_overlap) {
                lru_entries_.erase(list_it);
                map_it = address_map_.erase(map_it);
            } else {
                ++map_it;
            }
        }
    }

    // Try to create a new pinned region. On failure, evict the oldest entry that
    // shares an MMIO device with the target and retry. Repeat until success or
    // no evictable entries remain.
    while (true) {
        try {
            // PinnedMemory::Create also calls HostBufferSetPinnedMemory internally
            // as part of its public API. We immediately clear it since the cache is
            // the long-term owner; callers set it transiently when needed.
            auto pinned = PinnedMemory::Create(mesh_device, coordinate_range_set, host_buffer, map_to_noc);
            HostBufferSetPinnedMemory(host_buffer, nullptr);

            CacheEntry entry{pinned, host_addr, target_device_ids, target_mmio_ids, map_to_noc};
            lru_entries_.push_back(std::move(entry));
            address_map_.emplace(host_addr, std::prev(lru_entries_.end()));
            return pinned;
        } catch (...) {
            // Pin limit exceeded. Find the oldest LRU entry that shares an MMIO device
            // with the target and evict it to free a kernel pin slot.
            bool evicted = false;
            for (auto it = lru_entries_.begin(); it != lru_entries_.end(); ++it) {
                if (it->pinned_memory.use_count() > 1) {
                    continue;
                }
                for (ChipId mmio_id : it->mmio_device_ids) {
                    if (target_mmio_ids.contains(mmio_id)) {
                        erase_entry(it);
                        evicted = true;
                        break;
                    }
                }
                if (evicted) {
                    break;
                }
            }
            if (!evicted) {
                // No evictable entries on the relevant MMIO device; give up.
                return nullptr;
            }
            // Loop and retry.
        }
    }
}

void PinnedMemoryCache::release(const DistributedHostBuffer& buffer) {
    std::lock_guard<std::mutex> lock(mutex_);
    buffer.apply([this](const HostBuffer& shard) {
        const void* addr = static_cast<const void*>(shard.view_bytes().data());
        auto [range_begin, range_end] = address_map_.equal_range(addr);
        for (auto map_it = range_begin; map_it != range_end;) {
            lru_entries_.erase(map_it->second);
            map_it = address_map_.erase(map_it);
        }
    });
}

void PinnedMemoryCache::release_for_device(distributed::MeshDevice& mesh_device) {
    // Compute the set of MMIO device IDs for all chips in this MeshDevice.
    auto& cluster = MetalContext::instance().get_cluster();
    std::set<ChipId> mmio_device_ids;
    for (ChipId chip_id : mesh_device.get_device_ids()) {
        mmio_device_ids.insert(cluster.get_associated_mmio_device(chip_id));
    }

    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = lru_entries_.begin(); it != lru_entries_.end();) {
        bool overlaps = false;
        for (ChipId mmio_id : it->mmio_device_ids) {
            if (mmio_device_ids.contains(mmio_id)) {
                overlaps = true;
                break;
            }
        }
        if (overlaps) {
            auto next = std::next(it);
            erase_entry(it);
            it = next;
        } else {
            ++it;
        }
    }
}

}  // namespace tt::tt_metal::experimental
