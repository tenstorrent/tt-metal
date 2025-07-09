// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/distributed.hpp>

namespace tt::tt_metal {

class Buffer;
class IDevice;

namespace experimental {

class GlobalCircularBuffer {
public:
    GlobalCircularBuffer(
        IDevice* device,
        const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
        uint32_t size,
        BufferType buffer_type = BufferType::L1);

    GlobalCircularBuffer(const GlobalCircularBuffer&) = default;
    GlobalCircularBuffer& operator=(const GlobalCircularBuffer&) = default;

    GlobalCircularBuffer(GlobalCircularBuffer&&) noexcept = default;
    GlobalCircularBuffer& operator=(GlobalCircularBuffer&&) noexcept = default;

    const Buffer& cb_buffer() const;

    const CoreRangeSet& sender_cores() const;
    const CoreRangeSet& receiver_cores() const;
    const CoreRangeSet& all_cores() const;
    DeviceAddr buffer_address() const;
    DeviceAddr config_address() const;
    uint32_t size() const;
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping() const;
    IDevice* get_device() const { return this->device_; }

    static constexpr auto attribute_names = std::forward_as_tuple("sender_receiver_core_mapping", "size");
    auto attribute_values() const { return std::make_tuple(this->sender_receiver_core_mapping_, this->size_); }

private:
    void setup_cb_buffers(BufferType buffer_type, uint32_t max_num_receivers_per_sender);

    // GlobalCircularBuffer is implemented as a wrapper around a sharded buffer
    // This can be updated in the future to be its own container with optimized dispatch functions
    distributed::AnyBuffer cb_buffer_;
    distributed::AnyBuffer cb_config_buffer_;
    IDevice* device_;
    std::vector<std::pair<CoreCoord, CoreRangeSet>> sender_receiver_core_mapping_;
    CoreRangeSet sender_cores_;
    CoreRangeSet receiver_cores_;
    CoreRangeSet all_cores_;
    uint32_t size_ = 0;
};

/**
 * @brief Allocates a global circular buffer in L1 on the device.
 *
 * @param device The device to create the global circular buffer on.
 * @param sender_receiver_core_mapping The mapping of remote sender to remote receiver cores for the circular buffer.
 * @param size Size of the global circular buffer per core in bytes.
 * @param buffer_type Buffer type to store the global circular buffer. Can only be an L1 buffer type.\
 * @return The allocated global circular buffer.
 */
GlobalCircularBuffer CreateGlobalCircularBuffer(
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type = BufferType::L1);

/**
 * @brief Creates a Circular Buffer in L1 memory of specified cores using the address space of the
 * global circular bufferand adds it to the program.
 *
 * @param program The program to which the buffer will be added.
 * @param core_spec Specifies the cores where the circular buffer will be configured.
 * @param config Configuration for the circular buffer.
 * @param global_circular_buffer Global circular buffer to use the address space and configuration of.
 * @return CBHandle representing the Circular Buffer ID.
 */
CBHandle CreateCircularBuffer(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const CircularBufferConfig& config,
    const GlobalCircularBuffer& global_circular_buffer);

/**
 * @brief Updates the address of a dynamic global circular buffer.
 *
 * @param program The program containing the circular buffer.
 * @param cb_handle Handle of the circular buffer.
 * @param buffer Dynamically allocated global L1 buffer that shares address space with the circular buffer.
 */
void UpdateDynamicCircularBufferAddress(
    Program& program, CBHandle cb_handle, const GlobalCircularBuffer& global_circular_buffer);

}  // namespace experimental

}  // namespace tt::tt_metal

namespace std {

template <>
struct hash<tt::tt_metal::experimental::GlobalCircularBuffer> {
    std::size_t operator()(const tt::tt_metal::experimental::GlobalCircularBuffer& global_circular_buffer) const;
};

}  // namespace std
