// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace tt::umd {
class SysmemBuffer;
}

namespace tt::tt_metal {

class IDevice;
class PinnedMemoryImpl;

namespace distributed {
class MeshDevice;
}

using chip_id_t = int;

/**
 * @brief PinnedMemory manages system memory buffers across multiple devices.
 * 
 * This class provides a convenient wrapper around UMD SysmemBuffers, managing
 * one buffer per device in a mesh or set of devices. It handles allocation,
 * mapping, and access to pinned system memory that can be accessed by the devices.
 */
class PinnedMemory {
public:
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
    tt::umd::SysmemBuffer& get_buffer(chip_id_t device_id);
    const tt::umd::SysmemBuffer& get_buffer(chip_id_t device_id) const;

    /**
     * @brief Get host pointer for a specific device's buffer
     * @param device_id The device ID to get the host pointer for
     * @return Host pointer to the buffer
     */
    void* get_host_ptr(chip_id_t device_id);
    const void* get_host_ptr(chip_id_t device_id) const;

    /**
     * @brief Get device address for a specific device's buffer
     * @param device_id The device ID to get the device address for
     * @return Device address of the buffer
     */
    uint64_t get_device_addr(chip_id_t device_id) const;

    /**
     * @brief Get the buffer size per device
     * @return Size of each device's buffer in bytes
     */
    size_t get_buffer_size() const;

    /**
     * @brief Get the number of devices this PinnedMemory manages
     * @return Number of devices
     */
    size_t get_num_devices() const;

    /**
     * @brief Get all device IDs managed by this PinnedMemory
     * @return Vector of device IDs
     */
    std::vector<chip_id_t> get_device_ids() const;

    /**
     * @brief Check if a device ID is managed by this PinnedMemory
     * @param device_id The device ID to check
     * @return True if the device is managed, false otherwise
     */
    bool has_device(chip_id_t device_id) const;

    /**
     * @brief Write data to a specific device's buffer
     * @param device_id The device ID to write to
     * @param src Source data pointer
     * @param size Size of data to write
     * @param offset Offset within the device buffer to write to
     */
    void write_to_device(chip_id_t device_id, const void* src, size_t size, size_t offset = 0);

    /**
     * @brief Read data from a specific device's buffer
     * @param device_id The device ID to read from
     * @param dest Destination data pointer
     * @param size Size of data to read
     * @param offset Offset within the device buffer to read from
     */
    void read_from_device(chip_id_t device_id, void* dest, size_t size, size_t offset = 0);

private:
    friend class distributed::MeshDevice;

    /**
     * @brief Construct PinnedMemory by mapping existing host memory to devices
     * @param devices Vector of devices to map buffers for
     * @param host_buffer Existing host memory to map (must not be null)
     * @param buffer_size Size of buffer to map
     * @param map_to_noc Whether to map the buffer to the NOC
     */
    PinnedMemory(
        const std::vector<IDevice*>& devices,
        void* host_buffer,
        size_t buffer_size,
        bool map_to_noc = false);

    std::unique_ptr<PinnedMemoryImpl> pImpl;
};

}  // namespace tt::tt_metal 