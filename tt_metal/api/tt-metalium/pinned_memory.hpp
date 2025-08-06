// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>



namespace tt::umd {
class SysmemBuffer;
class SysmemManager;
}

namespace tt::tt_metal {

class IDevice;

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
    // MeshDevice is responsible for creating PinnedMemory instances
    friend class distributed::MeshDevice;

public:
    ~PinnedMemory();

    // Delete copy constructor and assignment operator
    PinnedMemory(const PinnedMemory&) = delete;
    PinnedMemory& operator=(const PinnedMemory&) = delete;

    // Move constructor and assignment operator
    PinnedMemory(PinnedMemory&& other) noexcept;
    PinnedMemory& operator=(PinnedMemory&& other) noexcept;

    /**
     * @brief Get the SysmemBuffer for a specific device
     * @param device_id The device ID to get the buffer for
     * @return Reference to the SysmemBuffer for the device
     */
    tt::umd::SysmemBuffer& get_buffer(chip_id_t device_id);

    /**
     * @brief Get the SysmemBuffer for a specific device (const version)
     * @param device_id The device ID to get the buffer for
     * @return Const reference to the SysmemBuffer for the device
     */
    const tt::umd::SysmemBuffer& get_buffer(chip_id_t device_id) const;

    /**
     * @brief Get host pointer for a specific device's buffer
     * @param device_id The device ID to get the pointer for
     * @return Host pointer to the buffer for the device
     */
    void* get_host_ptr(chip_id_t device_id);

    /**
     * @brief Get host pointer for a specific device's buffer (const version)
     * @param device_id The device ID to get the pointer for
     * @return Const host pointer to the buffer for the device
     */
    const void* get_host_ptr(chip_id_t device_id) const;

    /**
     * @brief Get device address for a specific device's buffer
     * @param device_id The device ID to get the address for
     * @return Device address of the buffer for the device
     */
    uint64_t get_device_addr(chip_id_t device_id) const;

    /**
     * @brief Get the buffer size per device
     * @return Size of each device's buffer in bytes
     */
    size_t get_buffer_size() const { return buffer_size_; }

    /**
     * @brief Get the number of devices this PinnedMemory manages
     * @return Number of devices
     */
    size_t get_num_devices() const { return device_buffers_.size(); }

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
    /**
     * @brief Construct PinnedMemory from a vector of devices
     * @param devices Vector of devices to allocate buffers for
     * @param buffer_size Size of buffer to allocate per device
     * @param map_to_noc Whether to map the buffer to the NOC
     */
    PinnedMemory(
        const std::vector<IDevice*>& devices,
        size_t buffer_size,
        bool map_to_noc = false);

    /**
     * @brief Construct PinnedMemory by mapping existing host memory to devices
     * @param devices Vector of devices to map buffers for
     * @param host_buffer Existing host memory to map (must be at least buffer_size * num_devices)
     * @param buffer_size Size of buffer per device
     * @param map_to_noc Whether to map the buffer to the NOC
     */
    PinnedMemory(
        const std::vector<IDevice*>& devices,
        void* host_buffer,
        size_t buffer_size,
        bool map_to_noc = false);

    void initialize_from_devices(
        const std::vector<IDevice*>& devices,
        void* host_buffer,
        size_t buffer_size,
        bool map_to_noc);

    size_t buffer_size_;
    bool map_to_noc_;
    bool owns_host_memory_;
    void* host_memory_base_;
    
    // Map from device ID to SysmemBuffer
    std::unordered_map<chip_id_t, std::unique_ptr<tt::umd::SysmemBuffer>> device_buffers_;
    
    // Map from device ID to SysmemManager (for cleanup and operations)
    std::unordered_map<chip_id_t, tt::umd::SysmemManager*> device_managers_;
};

}  // namespace tt::tt_metal 