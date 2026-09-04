// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <list>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>

#include <tt-metalium/experimental/pinned_memory.hpp>

namespace tt::tt_metal {
class HostBuffer;
namespace distributed {
class MeshDevice;
class MeshCoordinateRangeSet;
}  // namespace distributed
}  // namespace tt::tt_metal

namespace tt::tt_metal::experimental {

/**
 * @brief LRU cache for PinnedMemory objects, keyed by host buffer address.
 *
 * The cache is the long-term owner of all PinnedMemory objects used by the
 * tensor read/write paths. HostBuffer::pinned_memory_ is only populated
 * transiently (during enqueue_read calls) and cleared immediately afterward.
 *
 * The cache evicts LRU entries to keep total cached pinned memory under the
 * runtime-configured global limit, and cached memory per MMIO device under the
 * hardware NOC-mappable pin budget. When pinning fails due to the kernel pin
 * limit being exhausted, it evicts the oldest entry that shares an MMIO device
 * with the failing pin and retries.
 *
 * Two cleanup hooks ensure PinnedMemory never outlives its underlying memory
 * or its device:
 *  - HostBuffer pins release entries associated with their host address before
 *    the last pin releases the underlying host memory.
 *  - MeshDevice::~MeshDevice() calls release_for_device(mmio_ids) to remove
 *    all entries associated with the closing device.
 *
 * Callers must always handle a nullptr return from try_pin by falling back to
 * the unpinned transfer path.
 */
class PinnedMemoryCache {
public:
    static PinnedMemoryCache& instance();

    /**
     * @brief Try to obtain pinned memory for a HostBuffer.
     *
     * If the buffer is already cached (by host address), returns the existing
     * PinnedMemory immediately. Otherwise evicts global LRU entries to make
     * room under the cache-size limit, then creates a new one, retrying after
     * evicting LRU entries on the target MMIO device if creation fails.
     * Returns nullptr on permanent failure (not an error condition).
     *
     * @param mesh_device          Mesh device to pin for.
     * @param coordinate_range_set Set of device coordinates to pin for.
     * @param host_buffer          Host buffer whose memory should be pinned.
     * @param map_to_noc           Whether to map the buffer for direct NOC access.
     * @return Shared pointer to PinnedMemory, or nullptr if pinning is unavailable.
     */
    std::shared_ptr<PinnedMemory> try_pin(
        distributed::MeshDevice& mesh_device,
        const distributed::MeshCoordinateRangeSet& coordinate_range_set,
        HostBuffer& host_buffer,
        bool map_to_noc = false,
        PinnedMemoryDeviceAccess access = PinnedMemoryDeviceAccess::ReadWrite);

    /**
     * @brief Release all cache entries whose host address matches `host_address`.
     */
    void release(const void* host_address);

    /**
     * @brief Release all cache entries associated with the given MeshDevice.
     *
     * Called from MeshDevice::~MeshDevice() to ensure cached PinnedMemory
     * objects do not outlive the device they were created for. Internally
     * maps the device's chip IDs to MMIO device IDs and evicts all matching
     * entries.
     */
    void release_for_device(distributed::MeshDevice& mesh_device);

    /**
     * @brief Return the number of entries currently held in the cache.
     */
    size_t num_entries() const;

private:
    PinnedMemoryCache() = default;

    // ChipId = int (matches tt_metal/api/tt-metalium/mesh_config.hpp)
    struct CacheEntry {
        std::shared_ptr<PinnedMemory> pinned_memory;
        const void* host_address = nullptr;
        std::set<int> device_ids;
        std::set<int> mmio_device_ids;
        bool map_to_noc = false;
        PinnedMemoryDeviceAccess access = PinnedMemoryDeviceAccess::ReadWrite;
    };

    // Compute the set of chip IDs covered by the requested mesh coordinate range.
    std::set<int> compute_device_ids(
        distributed::MeshDevice& mesh_device, const distributed::MeshCoordinateRangeSet& coordinate_range_set);

    // Compute the set of MMIO chip IDs that back the requested device IDs.
    std::set<int> compute_mmio_device_ids(const std::set<int>& device_ids);

    // Erase the entry pointed to by `it` from both lru_entries_ and address_map_.
    void erase_entry(std::list<CacheEntry>::iterator it);

    // Erase all entries for `host_address`. Caller must hold mutex_.
    void release_locked(const void* host_address);

    // Evict oldest unreferenced entries until adding `incoming_size_bytes` keeps the global cache under
    // `global_limit_bytes` and each target MMIO device under `per_mmio_limit_bytes`.
    bool evict_oldest_until_within_limit(
        size_t incoming_size_bytes,
        size_t global_limit_bytes,
        size_t per_mmio_limit_bytes,
        const std::set<int>& target_mmio_ids);

    // Evict the oldest unreferenced entry that overlaps the target MMIO devices.
    bool evict_oldest_entry_for_mmio_ids(const std::set<int>& target_mmio_ids);

    // Evict the oldest unreferenced entry across all MMIO devices.
    bool evict_oldest_entry();

    // Check whether any existing entry for `host_addr` has MMIO overlap with
    // `target_mmio_ids` and is still referenced externally (use_count > 1),
    // which would prevent the kernel driver from accepting a new pin.
    bool has_conflicting_entry(const void* host_addr, const std::set<int>& target_mmio_ids) const;

    mutable std::mutex mutex_;
    std::list<CacheEntry> lru_entries_;  // front = oldest (LRU candidate)
    std::unordered_multimap<const void*, std::list<CacheEntry>::iterator> address_map_;
    size_t current_size_bytes_ = 0;
    std::unordered_map<int, size_t> current_size_bytes_by_mmio_id_;
};

}  // namespace tt::tt_metal::experimental
