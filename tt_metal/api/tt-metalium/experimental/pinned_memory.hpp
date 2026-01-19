// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <umd/device/types/cluster_descriptor_types.hpp>

namespace tt::umd {
class SysmemBuffer;
}

namespace tt::tt_metal {

class IDevice;
class HostBuffer;

namespace distributed {
class MeshDevice;
class MeshEvent;
class MeshCoordinateRangeSet;
class ShardDataTransfer;
}  // namespace distributed

namespace experimental {
class PinnedMemoryImpl;

struct MemoryPinningParameters {
    uint32_t max_pins;
    uint64_t max_total_pin_size;
    bool can_map_to_noc;
};

/**
 * @brief PinnedMemory manages system memory buffers across multiple devices.
 *
 * This class provides a convenient wrapper around UMD SysmemBuffers, managing
 * one buffer per device in a mesh or set of devices. It handles allocation,
 * mapping, and access to pinned system memory that can be accessed by the devices.
 */
class PinnedMemory {
public:
    /**
     * @brief NOC address and the MMIO device ID where it's usable from.
     *
     * This address may have a 64-bit offset, so it must be used with noc_wwrite_with_state and the variant of
     * noc_read_with_state that takes src_noc_addr as an argument.
     */
    struct NocAddr {
        uint32_t pcie_xy_enc;
        uint64_t addr;
        ChipId device_id;
    };

    ~PinnedMemory();

    // Move semantics
    PinnedMemory(PinnedMemory&& other) noexcept;
    PinnedMemory& operator=(PinnedMemory&& other) noexcept;

    // Delete copy semantics
    PinnedMemory(const PinnedMemory&) = delete;
    PinnedMemory& operator=(const PinnedMemory&) = delete;

    /**
     * @brief Get the underlying SysmemBuffer for a specific device
     * @param device_id The device ID to get the buffer for
     * @return Reference to the SysmemBuffer
     */
    tt::umd::SysmemBuffer& get_buffer(ChipId device_id);
    const tt::umd::SysmemBuffer& get_buffer(ChipId device_id) const;

    /**
     * @brief Get host pointer to the shared pinned memory
     * @return Host pointer to the buffer (same for all devices)
     */
    void* get_host_ptr();
    const void* get_host_ptr() const;

    /**
     * @brief Get device address for a specific device's buffer
     * @param device_id The device ID to get the device address for
     * @return Device address of the buffer
     */
    uint64_t get_device_addr(ChipId device_id) const;

    /**
     * @brief Get NOC address and the chip where it's usable from
     * @param device_id The device ID to get the NOC address for
     * @return Optional pair of (NOC address, MMIO chip ID) if buffer is mapped to NOC, nullopt otherwise
     */
    std::optional<NocAddr> get_noc_addr(ChipId device_id) const;

    /**
     * @brief Get the buffer size.
     * @return Size of the pinned region in bytes.
     */
    size_t get_buffer_size() const;

    /**
     * @brief Get all device IDs managed by this PinnedMemory
     * @return Vector of device IDs
     */
    std::vector<ChipId> get_device_ids() const;

    /**
     * @brief Check if a device ID is managed by this PinnedMemory
     * @param device_id The device ID to check
     * @return True if the device is managed, false otherwise
     */
    bool has_device(ChipId device_id) const;

    /**
     * @brief Check if the pinned memory is usable from NOC for a specific device
     * @param device_id The device ID to check
     * @return True if the device can access the buffer via NOC (i.e., map_to_noc is true and device is MMIO-capable)
     */
    bool usable_from_noc(ChipId device_id) const;

    /**
     * @brief Add a barrier event that must complete before memory can be locked
     * @param event The MeshEvent to add as a barrier
     *
     * This method adds an event to the barrier queue. The event must complete
     * before the memory can be safely accessed via lock().
     */
    void add_barrier_event(const distributed::MeshEvent& event);

    /**
     * @brief Lock the pinned memory for host access
     * @return Pointer to the host memory that can be safely accessed
     *
     * This method blocks until all pending barrier events have completed,
     * then returns a pointer to the pinned memory. The memory should be
     * unlocked with unlock() when access is no longer needed.
     */
    void* lock();

    /**
     * @brief Unlock the pinned memory
     *
     * This method releases the lock on the pinned memory, allowing it to be
     * safely accessed by devices again. Currently this is a no-op but should
     * be called for future compatibility and clarity of intent.
     */
    void unlock();

    /**
     * @brief Pin existing host memory for a specific set of mesh coordinates.
     *
     * Creates pinned system memory for the devices covered by the provided mesh coordinate range set,
     * mapping the existing host buffer to each device. When map_to_noc is true and supported by the
     * system, the buffer will be mapped to the NOC for direct device access.
     *
     * @param mesh_device The mesh device to pin memory for
     * @param coordinate_range_set Set of mesh coordinates to pin memory for
     * @param host_buffer Existing host memory to map (must not be null)
     * @param map_to_noc Whether to map the buffer to the NOC
     * @return Unique pointer to the created PinnedMemory instance
     */
    static std::unique_ptr<PinnedMemory> Create(
        distributed::MeshDevice& mesh_device,
        const distributed::MeshCoordinateRangeSet& coordinate_range_set,
        HostBuffer& host_buffer,
        bool map_to_noc = false);

private:
    friend class distributed::MeshDevice;

    /**
     * @brief Construct PinnedMemory by mapping existing host memory to devices
     * @param devices Vector of devices to map buffers for
     * @param host_buffer Existing host memory to map (must not be null)
     * @param buffer_size Size of buffer to map
     * @param map_to_noc Whether to map the buffer to the NOC
     */
    PinnedMemory(const std::vector<IDevice*>& devices, void* host_buffer, size_t buffer_size, bool map_to_noc = false);

    std::unique_ptr<PinnedMemoryImpl> pImpl;
};

/**
 * @brief Get the memory pinning parameters for a mesh device.
 * @param mesh_device The mesh device to get the memory pinning parameters for
 * @return Memory pinning parameters
 */
experimental::MemoryPinningParameters GetMemoryPinningParameters(distributed::MeshDevice& mesh_device);

std::shared_ptr<PinnedMemory> HostBufferGetPinnedMemory(HostBuffer& host_buffer);
void HostBufferSetPinnedMemory(HostBuffer& host_buffer, std::shared_ptr<PinnedMemory> pinned_memory);

const std::shared_ptr<PinnedMemory>& ShardDataTransferGetPinnedMemory(
    const distributed::ShardDataTransfer& shard_data_transfer);
void ShardDataTransferSetPinnedMemory(
    distributed::ShardDataTransfer& shard_data_transfer, std::shared_ptr<PinnedMemory> pinned_memory);

}  // namespace experimental

}  // namespace tt::tt_metal
