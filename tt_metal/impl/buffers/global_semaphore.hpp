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

class GlobalSemaphore {
public:
    GlobalSemaphore(
        Device* device,
        const CoreRangeSet& cores,
        uint32_t initial_value,
        BufferType buffer_type = BufferType::L1,
        tt::stl::Span<const SubDeviceId> sub_device_ids = {});

    GlobalSemaphore(
        Device* device,
        CoreRangeSet&& cores,
        uint32_t initial_value,
        BufferType buffer_type = BufferType::L1,
        tt::stl::Span<const SubDeviceId> sub_device_ids = {});

    GlobalSemaphore(const GlobalSemaphore&) = default;
    GlobalSemaphore& operator=(const GlobalSemaphore&) = default;

    GlobalSemaphore(GlobalSemaphore&&) noexcept = default;
    GlobalSemaphore& operator=(GlobalSemaphore&&) noexcept = default;

    Device* device() const;

    DeviceAddr address() const;

    void reset_semaphore_value(uint32_t reset_value, tt::stl::Span<const SubDeviceId> sub_device_ids = {}) const;

    static constexpr auto attribute_names = std::forward_as_tuple("cores");
    const auto attribute_values() const { return std::make_tuple(this->cores_); }

private:
    void setup_buffer(uint32_t initial_value, BufferType buffer_type, tt::stl::Span<const SubDeviceId> sub_device_ids);

    // GlobalSemaphore is implemented as a wrapper around a sharded buffer
    // This can be updated in the future to be its own container with optimized dispatch functions
    std::shared_ptr<Buffer> buffer_;
    Device* device_;
    CoreRangeSet cores_;
};

}  // namespace v0

}  // namespace tt::tt_metal

namespace std {

template <>
struct hash<tt::tt_metal::GlobalSemaphore> {
    std::size_t operator()(const tt::tt_metal::GlobalSemaphore& global_semaphore) const;
};

}  // namespace std
