// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pinned_memory_cache.hpp"

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <functional>
#include <istream>
#include <mutex>
#include <streambuf>
#include <string>
#include <string_view>
#include <sys/ioctl.h>
#include <unistd.h>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include "impl/context/metal_env_impl.hpp"
#include "llrt/tt_cluster.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "distributed/mesh_device_view_impl.hpp"

namespace tt::tt_metal::experimental {

using ChipId = int;

namespace {

// PROCMAP_QUERY (Linux 6.11, include/uapi/linux/fs.h), declared for older headers; its size is in the ioctl number.
struct ProcmapQuery {
    uint64_t size;
    uint64_t query_flags;
    uint64_t query_addr;
    uint64_t vma_start;
    uint64_t vma_end;
    uint64_t vma_flags;
    uint64_t vma_page_size;
    uint64_t vma_offset;
    uint64_t inode;
    uint32_t dev_major;
    uint32_t dev_minor;
    uint32_t vma_name_size;
    uint32_t build_id_size;
    uint64_t vma_name_addr;
    uint64_t build_id_addr;
};
constexpr unsigned long kProcmapQuery = _IOWR('f', 17, ProcmapQuery);
constexpr uint64_t kProcmapQueryVmaShared = 0x08;
constexpr uint64_t kProcmapQueryCoveringOrNextVma = 0x10;

// Reads a file descriptor as a stream, so the maps text can be parsed as it arrives; remembers a read error.
class FdReadBuffer : public std::streambuf {
public:
    explicit FdReadBuffer(int fd) : fd_(fd) {}
    bool failed() const { return failed_; }

protected:
    int_type underflow() override {
        while (true) {
            const ssize_t got = read(fd_, buffer_.data(), buffer_.size());
            if (got > 0) {
                setg(buffer_.data(), buffer_.data(), buffer_.data() + got);
                return traits_type::to_int_type(buffer_[0]);
            }
            if (got == 0 || errno != EINTR) {
                failed_ = got < 0;
                return traits_type::eof();
            }
        }
    }

private:
    int fd_;
    bool failed_ = false;
    std::array<char, 65536> buffer_{};
};

}  // namespace

PrivateFileMapping find_private_file_mapping_in_maps_text(std::istream& maps, uintptr_t begin, uintptr_t end) {
    uintptr_t covered = begin;  // [begin, covered) holds no private file mapping
    std::string line;
    while (covered < end && std::getline(maps, line)) {
        char* cursor = line.data();
        const uintptr_t vma_begin = std::strtoull(cursor, &cursor, 16);
        if (*cursor != '-') {
            continue;  // An unreadable line leaves a gap, which makes the answer Unknown below.
        }
        const uintptr_t vma_end = std::strtoull(cursor + 1, &cursor, 16);
        if (vma_end <= covered) {
            continue;
        }
        char perms[5] = {};
        unsigned long long inode = 0;
        if (vma_begin > covered || std::sscanf(cursor, " %4s %*x %*x:%*x %llu", perms, &inode) != 2) {
            return PrivateFileMapping::Unknown;
        }
        if (perms[3] == 'p' && inode != 0) {
            return PrivateFileMapping::Found;
        }
        covered = vma_end;
    }
    return covered >= end ? PrivateFileMapping::NotFound : PrivateFileMapping::Unknown;
}

PrivateFileMapping find_private_file_mapping(
    const void* address, size_t size, const char* maps_path, MapsSource* source) {
    MapsSource ignored = MapsSource::None;
    MapsSource& answered_by = source != nullptr ? *source : ignored;
    answered_by = MapsSource::None;
    if (size == 0) {
        return PrivateFileMapping::NotFound;
    }
    const auto begin = reinterpret_cast<uintptr_t>(address);
    const uintptr_t end = begin + size;
    if (end < begin) {
        return PrivateFileMapping::Unknown;
    }
    const int maps = open(maps_path, O_RDONLY | O_CLOEXEC);
    if (maps < 0) {
        return PrivateFileMapping::Unknown;
    }
    ProcmapQuery query{};
    query.size = sizeof(query);
    query.query_flags = kProcmapQueryCoveringOrNextVma;
    auto result = PrivateFileMapping::NotFound;
    bool use_text = false;
    answered_by = MapsSource::Ioctl;
    for (uintptr_t cursor = begin; cursor < end; cursor = query.vma_end) {
        query.query_addr = cursor;
        if (ioctl(maps, kProcmapQuery, &query) != 0) {
            use_text = errno != ENOENT;  // ENOENT: no VMA at or after the cursor. Otherwise: no PROCMAP_QUERY.
            result = PrivateFileMapping::Unknown;
            break;
        }
        if (query.vma_start > cursor || query.vma_end <= cursor) {  // a gap in the range, or no progress
            result = PrivateFileMapping::Unknown;
            break;
        }
        if ((query.vma_flags & kProcmapQueryVmaShared) == 0 && query.inode != 0) {
            result = PrivateFileMapping::Found;
            break;
        }
    }
    if (use_text) {
        answered_by = MapsSource::Text;
        FdReadBuffer buffer(maps);
        std::istream text(&buffer);
        result = find_private_file_mapping_in_maps_text(text, begin, end);
        if (buffer.failed()) {
            result = PrivateFileMapping::Unknown;
        }
    }
    close(maps);
    return result;
}

bool read_only_pin_allowed(
    const void* address,
    size_t size,
    PrivateFileMapping mapping,
    std::once_flag& unknown_warning,
    const std::function<void(std::string_view)>& warn) {
    if (mapping == PrivateFileMapping::NotFound) {
        return true;
    }
    if (mapping == PrivateFileMapping::Unknown) {
        std::call_once(unknown_warning, [&warn] {
            constexpr std::string_view warning =
                "PinnedMemoryCache: this thread's memory maps could not rule out a private file mapping (they were "
                "unreadable, incomplete or unparsable), so read-only pins are declined and those uploads are copied";
            if (warn) {
                warn(warning);
            } else {
                log_warning(tt::LogMetal, "{}", warning);
            }
        });
    }
    log_debug(
        tt::LogMetal,
        "PinnedMemoryCache: not pinning {} bytes at {} read-only: {}",
        size,
        address,
        mapping == PrivateFileMapping::Found
            ? "it overlaps a private file mapping, whose read-only pin can stall in the kernel"
            : "the thread's maps cannot rule out a private file mapping");
    return false;
}

bool read_only_pin_allowed(
    const void* address, size_t size, PrivateFileMapping mapping, const std::function<void(std::string_view)>& warn) {
    static std::once_flag unknown_warning;
    return read_only_pin_allowed(address, size, mapping, unknown_warning, warn);
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

    std::unique_lock<std::mutex> lock(mutex_);

    // Search all entries for this host address for a usable cache hit.
    auto find_cached = [&]() -> std::shared_ptr<PinnedMemory> {
        auto [range_begin, range_end] = address_map_.equal_range(host_addr);
        for (auto map_it = range_begin; map_it != range_end; ++map_it) {
            auto list_it = map_it->second;
            if (list_it->pinned_memory->get_buffer_size() < buffer_size) {
                continue;
            }
            if (map_to_noc && !list_it->map_to_noc) {
                continue;
            }
            if (access == PinnedMemoryDeviceAccess::ReadWrite &&
                list_it->access == PinnedMemoryDeviceAccess::ReadOnly) {
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
        return nullptr;
    };
    if (auto cached = find_cached()) {
        return cached;
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

    // A device-read-only pin of a private file mapping can loop forever in the kernel (#63423), so such memory is
    // copied. The check reads the thread's maps, so it runs unlocked; another thread may pin this buffer meanwhile.
    if (actual_access == PinnedMemoryDeviceAccess::ReadOnly) {
        if (buffer_size > global_cache_limit || buffer_size > per_mmio_pin_limit) {
            return nullptr;  // evict_oldest_until_within_limit would refuse it; skip the maps query
        }
        lock.unlock();
        const auto mapping = find_private_file_mapping(host_addr, buffer_size);
        lock.lock();
        if (!read_only_pin_allowed(host_addr, buffer_size, mapping)) {
            return nullptr;
        }
        if (auto cached = find_cached()) {
            return cached;
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
