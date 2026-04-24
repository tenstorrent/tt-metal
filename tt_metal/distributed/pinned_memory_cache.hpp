// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <list>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>

namespace tt::tt_metal {
class HostBuffer;
class DistributedHostBuffer;
namespace distributed {
class MeshDevice;
class MeshCoordinateRangeSet;
}  // namespace distributed
namespace experimental {
class PinnedMemory;
}  // namespace experimental
}  // namespace tt::tt_metal

namespace tt::tt_metal::experimental {

/**
 * @brief LRU cache for PinnedMemory objects, keyed by host buffer address.
 *
 * The cache is the long-term owner of all PinnedMemory objects used by the
 * tensor read/write paths. HostBuffer::pinned_memory_ is only populated
 * transiently (during enqueue_read calls) and cleared immediately afterward.
 *
 * When pinning fails due to the kernel pin limit being exhausted, the cache
 * evicts the oldest entry that shares an MMIO device with the failing pin and
 * retries, repeating until either success or no evictable entries remain.
 *
 * Two cleanup hooks ensure PinnedMemory never outlives its underlying memory
 * or its device:
 *  - HostStorage::~HostStorage() calls release(buffer) to remove all entries
 *    associated with the destroyed host buffers.
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
     * PinnedMemory immediately. Otherwise creates a new one, retrying after
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
        bool map_to_noc = false);

    /**
     * @brief Release all cache entries whose host address matches a shard
     *        in the given DistributedHostBuffer.
     *
     * Called from HostStorage::~HostStorage() to ensure cached PinnedMemory
     * objects do not outlive the host memory they pin.
     */
    void release(const DistributedHostBuffer& buffer);

    /**
     * @brief Release all cache entries associated with the given MeshDevice.
     *
     * Called from MeshDevice::~MeshDevice() to ensure cached PinnedMemory
     * objects do not outlive the device they were created for. Internally
     * maps the device's chip IDs to MMIO device IDs and evicts all matching
     * entries.
     */
    void release_for_device(distributed::MeshDevice& mesh_device);

private:
    PinnedMemoryCache() = default;

    // ChipId = int (matches tt_metal/api/tt-metalium/mesh_config.hpp)
    struct CacheEntry {
        std::shared_ptr<PinnedMemory> pinned_memory;
        const void* host_address = nullptr;
        std::set<int> device_ids;
        std::set<int> mmio_device_ids;
        bool map_to_noc = false;
    };

    // Compute the set of chip IDs covered by the requested mesh coordinate range.
    std::set<int> compute_device_ids(
        distributed::MeshDevice& mesh_device, const distributed::MeshCoordinateRangeSet& coordinate_range_set);

    // Compute the set of MMIO chip IDs that back the requested device IDs.
    std::set<int> compute_mmio_device_ids(const std::set<int>& device_ids);

    // Erase the entry pointed to by `it` from both lru_entries_ and address_map_.
    void erase_entry(std::list<CacheEntry>::iterator it);

    // Check whether any existing entry for `host_addr` has MMIO overlap with
    // `target_mmio_ids` and is still referenced externally (use_count > 1),
    // which would prevent the kernel driver from accepting a new pin.
    bool has_conflicting_entry(const void* host_addr, const std::set<int>& target_mmio_ids) const;

    std::mutex mutex_;
    std::list<CacheEntry> lru_entries_;  // front = oldest (LRU candidate)
    std::unordered_multimap<const void*, std::list<CacheEntry>::iterator> address_map_;
};

}  // namespace tt::tt_metal::experimental
