// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>

#include "core_coord.hpp"
#include "buffer_constants.hpp"
#include "hal.hpp"

namespace tt::tt_metal {

inline namespace v0 {

class Buffer;
class IDevice;

class GlobalSemaphore {
public:
    GlobalSemaphore(
        IDevice* device, const CoreRangeSet& cores, uint32_t initial_value, BufferType buffer_type = BufferType::L1);

    GlobalSemaphore(
        IDevice* device, CoreRangeSet&& cores, uint32_t initial_value, BufferType buffer_type = BufferType::L1);

    GlobalSemaphore(const GlobalSemaphore&) = default;
    GlobalSemaphore& operator=(const GlobalSemaphore&) = default;

    GlobalSemaphore(GlobalSemaphore&&) noexcept = default;
    GlobalSemaphore& operator=(GlobalSemaphore&&) noexcept = default;

    IDevice* device() const;

    DeviceAddr address() const;

    void reset_semaphore_value(uint32_t reset_value) const;

    static constexpr auto attribute_names = std::forward_as_tuple("cores");
    const auto attribute_values() const { return std::make_tuple(this->cores_); }

private:
    void setup_buffer(uint32_t initial_value, BufferType buffer_type);

    // GlobalSemaphore is implemented as a wrapper around a sharded buffer
    // This can be updated in the future to be its own container with optimized dispatch functions
    std::shared_ptr<Buffer> buffer_;
    IDevice* device_;
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
