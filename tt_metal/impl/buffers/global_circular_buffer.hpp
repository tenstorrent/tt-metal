// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>

#include "tt_metal/common/core_coord.hpp"
#include "tt_metal/impl/buffers/buffer_constants.hpp"
#include "tt_metal/impl/sub_device/sub_device_types.hpp"
#include "tt_metal/llrt/hal.hpp"

namespace tt::tt_metal {

inline namespace v0 {

class Buffer;
class Device;

}  // namespace v0

namespace v1 {

namespace experimental {

class GlobalCircularBuffer {
public:
    GlobalCircularBuffer(
        Device* device,
        const std::unordered_map<CoreCoord, CoreRangeSet>& sender_receiver_core_mapping,
        uint32_t size,
        BufferType buffer_type = BufferType::L1,
        tt::stl::Span<const SubDeviceId> sub_device_ids = {});

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

    static constexpr auto attribute_names = std::forward_as_tuple("sender_receiver_core_mapping", "size");
    const auto attribute_values() const { return std::make_tuple(this->sender_receiver_core_mapping_, this->size_); }

private:
    void setup_cb_buffers(
        BufferType buffer_type, uint32_t max_num_receivers_per_sender, tt::stl::Span<const SubDeviceId> sub_device_ids);

    // GlobalCircularBuffer is implemented as a wrapper around a sharded buffer
    // This can be updated in the future to be its own container with optimized dispatch functions
    std::shared_ptr<Buffer> cb_buffer_;
    std::shared_ptr<Buffer> cb_config_buffer_;
    Device* device_;
    std::unordered_map<CoreCoord, CoreRangeSet> sender_receiver_core_mapping_;
    CoreRangeSet sender_cores_;
    CoreRangeSet receiver_cores_;
    CoreRangeSet all_cores_;
    uint32_t size_ = 0;
};

}  // namespace experimental

}  // namespace v1

}  // namespace tt::tt_metal

namespace std {

template <>
struct hash<tt::tt_metal::v1::experimental::GlobalCircularBuffer> {
    std::size_t operator()(const tt::tt_metal::v1::experimental::GlobalCircularBuffer& global_circular_buffer) const;
};

}  // namespace std
