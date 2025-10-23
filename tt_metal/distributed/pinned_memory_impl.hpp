// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>
#include <deque>

#include <tt-metalium/pinned_memory.hpp>

namespace tt::umd {
class SysmemBuffer;
}

namespace tt::tt_metal {

class IDevice;
using ChipId = int;

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
        const std::vector<IDevice*>& devices, void* host_buffer, size_t buffer_size, bool map_to_noc = false);

    ~PinnedMemoryImpl();

    // Move semantics
    PinnedMemoryImpl(PinnedMemoryImpl&& other) noexcept;
    PinnedMemoryImpl& operator=(PinnedMemoryImpl&& other) noexcept;

    // Delete copy semantics
    PinnedMemoryImpl(const PinnedMemoryImpl&) = delete;
    PinnedMemoryImpl& operator=(const PinnedMemoryImpl&) = delete;

    // Buffer access methods
    tt::umd::SysmemBuffer& get_buffer(ChipId device_id);
    const tt::umd::SysmemBuffer& get_buffer(ChipId device_id) const;

    // Host pointer access methods
    void* get_host_ptr();
    const void* get_host_ptr() const;

    // Device address access method
    uint64_t get_device_addr(ChipId device_id) const;

    // NOC address access method
    std::optional<PinnedMemory::NocAddr> get_noc_addr(ChipId device_id) const;

    // Utility methods
    size_t get_buffer_size() const { return buffer_size_; }
    std::vector<ChipId> get_device_ids() const;
    bool has_device(ChipId device_id) const;
    bool usable_from_noc(ChipId device_id) const;

    void add_barrier_event(const distributed::MeshEvent& event);

    void* lock();

    void unlock();

private:
    void initialize_from_devices(
        const std::vector<IDevice*>& devices, void* host_buffer, size_t buffer_size, bool map_to_noc);

    size_t buffer_size_;
    bool map_to_noc_;
    // Offset from the aligned mapped base to the actual host buffer start
    size_t host_offset_ = 0;

    // Map from device ID to SysmemBuffer (keyed by MMIO device ID)
    std::unordered_map<ChipId, std::unique_ptr<tt::umd::SysmemBuffer>> device_buffers_;

    // Map from logical device ID to its associated MMIO device ID
    std::unordered_map<ChipId, ChipId> device_to_mmio_map_;

    std::deque<distributed::MeshEvent> barrier_events_;
};

}  // namespace tt::tt_metal
