// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pinned_memory_cache.hpp"

#include <sys/mman.h>
#include <linux/mman.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <thread>
#include <unordered_map>

#include <tt-logger/tt-logger.hpp>

#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include "impl/context/metal_env_impl.hpp"
#include "llrt/tt_cluster.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "distributed/mesh_device_view_impl.hpp"

namespace tt::tt_metal::experimental {

using ChipId = int;

namespace {

constexpr size_t k_staging_align = 4096;
constexpr size_t k_staging_copy_threads = 8;
constexpr size_t k_staging_parallel_min = 64ull << 20;

bool staging_enabled() {
    static const bool enabled = [] {
        const char* e = std::getenv("TT_METAL_PIN_STAGING");
        return e == nullptr || std::string_view(e) != "0";
    }();
    return enabled;
}

size_t staging_ring_bytes() {
    static const size_t bytes = [] {
        const char* e = std::getenv("TT_METAL_PIN_STAGING_MB");
        size_t mb = 1024;
        if (e != nullptr && *e != '\0') {
            mb = std::strtoull(e, nullptr, 10);
        }
        return tt::round_up(mb << 20, 2ull << 20);
    }();
    return bytes;
}

// memcpy into the ring on several threads: one thread copies at about half the DMA rate.
void staging_copy(std::byte* dst, const std::byte* src, size_t n) {
    if (n < k_staging_parallel_min) {
        std::memcpy(dst, src, n);
        return;
    }
    const size_t part = tt::round_up((n + k_staging_copy_threads - 1) / k_staging_copy_threads, k_staging_align);
    std::vector<std::thread> threads;
    threads.reserve(k_staging_copy_threads);
    for (size_t off = part; off < n; off += part) {
        const size_t len = std::min(part, n - off);
        threads.emplace_back([=] { std::memcpy(dst + off, src + off, len); });
    }
    std::memcpy(dst, src, std::min(part, n));
    for (auto& t : threads) {
        t.join();
    }
}

}  // namespace

PinnedMemoryCache::StagingRing::~StagingRing() {
    if (writes > 0 || too_big > 0) {
        log_info(
            tt::LogMetal,
            "pin staging: {} writes, {:.2f} GB staged, copy {:.3f} s, {} writes too big for the {} MiB ring",
            writes,
            static_cast<double>(bytes) / 1e9,
            copy_s,
            too_big,
            size >> 20);
    }
    pinned.reset();
    if (base != nullptr) {
        munmap(base, size);
    }
}

PinnedMemoryCache::StagingRing* PinnedMemoryCache::get_staging_ring(distributed::MeshDevice& mesh_device) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (auto it = staging_rings_.find(&mesh_device); it != staging_rings_.end()) {
        return it->second.get();
    }
    if (staging_unavailable_.contains(&mesh_device)) {
        return nullptr;
    }
    const auto params = GetMemoryPinningParameters(mesh_device);
    const size_t size = staging_ring_bytes();
    if (params.max_pins == 0 || !params.can_map_to_noc || size == 0) {
        staging_unavailable_.insert(&mesh_device);
        return nullptr;
    }

    // 1 GiB pages where the ring is a whole number of them and some are free, else 2 MiB pages.
    void* base = MAP_FAILED;
    std::string_view page = "1 GiB";
    if (size % (1ull << 30) == 0) {
        base = mmap(
            nullptr,
            size,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_1GB | MAP_POPULATE,
            -1,
            0);
    }
    if (base == MAP_FAILED) {
        page = "2 MiB";
        base = mmap(
            nullptr,
            size,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB | MAP_POPULATE,
            -1,
            0);
    }
    if (base == MAP_FAILED) {
        log_info(
            tt::LogMetal,
            "pin staging: no free hugepages for a {} MiB ring ({}); weights upload on the per-tensor pin path",
            size >> 20,
            std::strerror(errno));
        staging_unavailable_.insert(&mesh_device);
        return nullptr;
    }

    const auto& view = mesh_device.get_view();
    distributed::MeshCoordinateRangeSet local_range;
    size_t local_chips = 0;
    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device.shape())) {
        if (view.impl().is_local(coord)) {
            local_range.merge(distributed::MeshCoordinateRange(coord, coord));
            ++local_chips;
        }
    }

    auto ring = std::make_unique<StagingRing>();
    ring->base = base;
    ring->size = size;
    const auto t0 = std::chrono::steady_clock::now();
    try {
        HostBuffer host_buffer(ttsl::Span<std::byte>(static_cast<std::byte*>(base), size), MemoryPin());
        ring->pinned = PinnedMemory::Create(
            mesh_device,
            local_range,
            host_buffer,
            /*map_to_noc=*/true,
            params.supports_read_only ? PinnedMemoryDeviceAccess::ReadOnly : PinnedMemoryDeviceAccess::ReadWrite);
        HostBufferSetPinnedMemory(host_buffer, nullptr);
    } catch (const std::exception& e) {
        log_info(
            tt::LogMetal,
            "pin staging: pinning the {} MiB ring failed ({}); weights upload on the per-tensor pin path",
            size >> 20,
            e.what());
        staging_unavailable_.insert(&mesh_device);
        return nullptr;  // ring's destructor unmaps
    }
    const double pin_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    log_info(
        tt::LogMetal,
        "pin staging: {} MiB ring on {} pages, pinned once for {} chips in {:.3f} s",
        size >> 20,
        page,
        local_chips,
        pin_s);
    auto* raw = ring.get();
    staging_rings_.emplace(&mesh_device, std::move(ring));
    return raw;
}

std::optional<StagedWrite> PinnedMemoryCache::stage_for_write(
    distributed::MeshDevice& mesh_device, const std::vector<StagingSource>& sources) {
    if (!staging_enabled() || sources.empty()) {
        return std::nullopt;
    }
    StagingRing* ring = get_staging_ring(mesh_device);
    if (ring == nullptr) {
        return std::nullopt;
    }

    // One ring slot per distinct source (a replicated weight is copied once for every chip).
    std::unordered_map<const void*, size_t> offsets;
    size_t total = 0;
    for (const auto& src : sources) {
        if (offsets.emplace(src.data, total).second) {
            total += tt::round_up(src.size, k_staging_align);
        }
    }

    StagedWrite staged{std::unique_lock<std::mutex>(ring->mutex), {}};
    if (total > ring->size) {
        ++ring->too_big;
        return std::nullopt;
    }

    auto* base = static_cast<std::byte*>(ring->base);
    const auto t0 = std::chrono::steady_clock::now();
    for (const auto& [data, off] : offsets) {
        size_t size = 0;
        for (const auto& src : sources) {
            if (src.data == data) {
                size = src.size;
                break;
            }
        }
        staging_copy(base + off, static_cast<const std::byte*>(data), size);
        ring->bytes += size;
    }
    ring->copy_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    ++ring->writes;

    staged.transfers.reserve(sources.size());
    for (const auto& src : sources) {
        auto xfer = distributed::ShardDataTransfer{src.coord}
                        .host_data(static_cast<void*>(base + offsets.at(src.data)))
                        .region(BufferRegion(0, src.size));
        ShardDataTransferSetPinnedMemory(xfer, ring->pinned);
        staged.transfers.push_back(std::move(xfer));
    }
    return staged;
}

PinnedMemoryCache& PinnedMemoryCache::instance() {
    static PinnedMemoryCache cache;
    return cache;
}

size_t PinnedMemoryCache::num_entries() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return lru_entries_.size();
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

std::set<ChipId> PinnedMemoryCache::compute_mmio_device_ids(
    const tt::Cluster& cluster, const std::set<ChipId>& device_ids) {
    std::set<ChipId> mmio_ids;
    for (ChipId device_id : device_ids) {
        mmio_ids.insert(cluster.get_associated_mmio_device(device_id));
    }
    return mmio_ids;
}

void PinnedMemoryCache::erase_entry(std::list<CacheEntry>::iterator it) {
    const size_t entry_size_bytes = it->pinned_memory->get_buffer_size();
    auto [range_begin, range_end] = address_map_.equal_range(it->host_address);
    for (auto map_it = range_begin; map_it != range_end; ++map_it) {
        if (map_it->second == it) {
            address_map_.erase(map_it);
            break;
        }
    }
    for (ChipId mmio_id : it->mmio_device_ids) {
        auto current_size_it = current_size_bytes_by_mmio_id_.find(mmio_id);
        if (current_size_it == current_size_bytes_by_mmio_id_.end()) {
            continue;
        }
        current_size_it->second =
            entry_size_bytes <= current_size_it->second ? current_size_it->second - entry_size_bytes : size_t{0};
        if (current_size_it->second == 0) {
            current_size_bytes_by_mmio_id_.erase(current_size_it);
        }
    }
    current_size_bytes_ =
        entry_size_bytes <= current_size_bytes_ ? current_size_bytes_ - entry_size_bytes : static_cast<size_t>(0);
    lru_entries_.erase(it);
}

void PinnedMemoryCache::release_locked(const void* host_address) {
    auto [range_begin, range_end] = address_map_.equal_range(host_address);
    for (auto map_it = range_begin; map_it != range_end;) {
        auto list_it = map_it->second;
        auto next_map_it = std::next(map_it);
        erase_entry(list_it);
        map_it = next_map_it;
    }
}

bool PinnedMemoryCache::evict_oldest_entry_for_mmio_ids(const std::set<ChipId>& target_mmio_ids) {
    for (auto it = lru_entries_.begin(); it != lru_entries_.end(); ++it) {
        if (it->pinned_memory.use_count() > 1) {
            continue;
        }
        for (ChipId mmio_id : it->mmio_device_ids) {
            if (target_mmio_ids.contains(mmio_id)) {
                erase_entry(it);
                return true;
            }
        }
    }
    return false;
}

bool PinnedMemoryCache::evict_oldest_entry() {
    for (auto it = lru_entries_.begin(); it != lru_entries_.end(); ++it) {
        if (it->pinned_memory.use_count() > 1) {
            continue;
        }
        erase_entry(it);
        return true;
    }
    return false;
}

bool PinnedMemoryCache::evict_oldest_until_within_limit(
    size_t incoming_size_bytes,
    size_t global_limit_bytes,
    size_t per_mmio_limit_bytes,
    const std::set<ChipId>& target_mmio_ids) {
    if (incoming_size_bytes > global_limit_bytes || incoming_size_bytes > per_mmio_limit_bytes) {
        return false;
    }

    while (current_size_bytes_ > global_limit_bytes - incoming_size_bytes) {
        if (!evict_oldest_entry()) {
            return false;
        }
    }

    for (ChipId mmio_id : target_mmio_ids) {
        auto current_size_bytes = [&]() {
            auto current_size_it = current_size_bytes_by_mmio_id_.find(mmio_id);
            return current_size_it == current_size_bytes_by_mmio_id_.end() ? size_t{0} : current_size_it->second;
        };
        const std::set<ChipId> single_mmio_id{mmio_id};
        while (current_size_bytes() > per_mmio_limit_bytes - incoming_size_bytes) {
            if (!evict_oldest_entry_for_mmio_ids(single_mmio_id)) {
                return false;
            }
        }
    }
    return true;
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
    bool map_to_noc,
    PinnedMemoryDeviceAccess access) {
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
    auto& metal_env = mesh_device.impl().metal_env();
    const size_t global_cache_limit = metal_env.get_rtoptions().get_pinned_memory_cache_limit_bytes();
    const size_t per_mmio_pin_limit = params.max_total_pin_size;
    std::set<ChipId> target_device_ids = compute_device_ids(mesh_device, coordinate_range_set);
    if (target_device_ids.empty()) {
        return nullptr;
    }

    std::set<ChipId> target_mmio_ids = compute_mmio_device_ids(metal_env.get_cluster(), target_device_ids);

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
        if (access == PinnedMemoryDeviceAccess::ReadWrite && list_it->access == PinnedMemoryDeviceAccess::ReadOnly) {
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

    // Older KMDs cannot create a device-read-only mapping. Widen cache-created mappings to read/write so callers keep
    // using the pinned fast path; the cache records the mapping's actual permissions.
    const PinnedMemoryDeviceAccess actual_access =
        access == PinnedMemoryDeviceAccess::ReadOnly && !params.supports_read_only ? PinnedMemoryDeviceAccess::ReadWrite
                                                                                   : access;
    // A widened request asks the driver to pin for writing memory the caller only promised to read. When that memory is
    // genuinely non-writable -- an mmap(PROT_READ) of a file, the case read-only pinning exists for -- the pin fails
    // every time, no matter how many slots are free. Eviction cannot help, so such a failure must not drive the
    // retry loop below or it would unpin the entire cache for this device on the way to giving up.
    const bool widened_to_read_write = actual_access != access;

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
                auto next_map_it = std::next(map_it);
                erase_entry(list_it);
                map_it = next_map_it;
            } else {
                ++map_it;
            }
        }
    }

    if (!evict_oldest_until_within_limit(buffer_size, global_cache_limit, per_mmio_pin_limit, target_mmio_ids)) {
        return nullptr;
    }

    // Try to create a new pinned region. On failure, evict the oldest entry that
    // shares an MMIO device with the target and retry. Repeat until success or
    // no evictable entries remain.
    while (true) {
        try {
            // PinnedMemory::Create also calls HostBufferSetPinnedMemory internally
            // as part of its public API. We immediately clear it since the cache is
            // the long-term owner; callers set it transiently when needed.
            auto pinned =
                PinnedMemory::Create(mesh_device, coordinate_range_set, host_buffer, map_to_noc, actual_access);
            HostBufferSetPinnedMemory(host_buffer, nullptr);

            CacheEntry entry{pinned, host_addr, target_device_ids, target_mmio_ids, map_to_noc, actual_access};
            lru_entries_.push_back(std::move(entry));
            address_map_.emplace(host_addr, std::prev(lru_entries_.end()));
            for (ChipId mmio_id : target_mmio_ids) {
                current_size_bytes_by_mmio_id_[mmio_id] += pinned->get_buffer_size();
            }
            current_size_bytes_ += pinned->get_buffer_size();
            return pinned;
        } catch (...) {
            // A widened read-only request may be unpinnable for write no matter what is evicted; give up rather
            // than draining the cache. See widened_to_read_write above.
            if (widened_to_read_write) {
                return nullptr;
            }
            // Pin limit exceeded. Find the oldest LRU entry that shares an MMIO device
            // with the target and evict it to free a kernel pin slot.
            if (!evict_oldest_entry_for_mmio_ids(target_mmio_ids)) {
                // No evictable entries on the relevant MMIO device; give up.
                return nullptr;
            }
            // Loop and retry.
        }
    }
}

void PinnedMemoryCache::release(const void* host_address) {
    std::lock_guard<std::mutex> lock(mutex_);
    release_locked(host_address);
}

void PinnedMemoryCache::release_for_device(distributed::MeshDevice& mesh_device) {
    {
        // Unpin the staging ring while the mesh's chips are still open.
        std::unique_ptr<StagingRing> ring;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (auto it = staging_rings_.find(&mesh_device); it != staging_rings_.end()) {
                ring = std::move(it->second);
                staging_rings_.erase(it);
            }
            staging_unavailable_.erase(&mesh_device);
        }
    }

    // Compute the set of MMIO device IDs for all chips in this MeshDevice.
    auto& cluster = mesh_device.impl().metal_env().get_cluster();
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
