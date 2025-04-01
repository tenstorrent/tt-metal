// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer_constants.hpp>
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
    const auto attribute_values() const { return std::make_tuple(this->sender_receiver_core_mapping_, this->size_); }

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

}  // namespace experimental

}  // namespace tt::tt_metal

namespace std {

template <>
struct hash<tt::tt_metal::experimental::GlobalCircularBuffer> {
    std::size_t operator()(const tt::tt_metal::experimental::GlobalCircularBuffer& global_circular_buffer) const;
};

}  // namespace std
