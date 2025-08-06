// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tt::umd {
class SysmemBuffer;
}

namespace tt::tt_metal {

class IDevice;
using chip_id_t = int;

/**
 * @brief Implementation class for PinnedMemory using pimpl pattern
 * 
 * This class contains all the implementation details that were previously
 * in the public PinnedMemory interface.
 */
class PinnedMemoryImpl {
public:
    /**
     * @brief Construct PinnedMemory implementation from devices with existing host memory
     * @param devices Vector of devices to map buffers for
     * @param host_buffer Existing host memory to map (must not be null)
     * @param buffer_size Size of buffer to map
     * @param map_to_noc Whether to map the buffer to the NOC
     */
    PinnedMemoryImpl(
        const std::vector<IDevice*>& devices,
        void* host_buffer,
        size_t buffer_size,
        bool map_to_noc = false);

    ~PinnedMemoryImpl();

    // Move semantics
    PinnedMemoryImpl(PinnedMemoryImpl&& other) noexcept;
    PinnedMemoryImpl& operator=(PinnedMemoryImpl&& other) noexcept;

    // Delete copy semantics
    PinnedMemoryImpl(const PinnedMemoryImpl&) = delete;
    PinnedMemoryImpl& operator=(const PinnedMemoryImpl&) = delete;

    // Buffer access methods
    tt::umd::SysmemBuffer& get_buffer(chip_id_t device_id);
    const tt::umd::SysmemBuffer& get_buffer(chip_id_t device_id) const;

    // Host pointer access methods
    void* get_host_ptr(chip_id_t device_id);
    const void* get_host_ptr(chip_id_t device_id) const;

    // Device address access method
    uint64_t get_device_addr(chip_id_t device_id) const;

    // Utility methods
    size_t get_buffer_size() const { return buffer_size_; }
    size_t get_num_devices() const { return device_to_mmio_map_.size(); }
    std::vector<chip_id_t> get_device_ids() const;
    bool has_device(chip_id_t device_id) const;

    // Data transfer methods
    void write_to_device(chip_id_t device_id, const void* src, size_t size, size_t offset = 0);
    void read_from_device(chip_id_t device_id, void* dest, size_t size, size_t offset = 0);

private:
    void initialize_from_devices(
        const std::vector<IDevice*>& devices,
        void* host_buffer,
        size_t buffer_size,
        bool map_to_noc);

    size_t buffer_size_;
    bool map_to_noc_;
    bool owns_host_memory_;
    void* host_memory_base_;
    
    // Map from device ID to SysmemBuffer (keyed by MMIO device ID)
    std::unordered_map<chip_id_t, std::unique_ptr<tt::umd::SysmemBuffer>> device_buffers_;
    
    // Map from logical device ID to its associated MMIO device ID
    std::unordered_map<chip_id_t, chip_id_t> device_to_mmio_map_;
};

}  // namespace tt::tt_metal