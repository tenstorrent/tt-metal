// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <tuple>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>

// forward declarations
namespace tt::tt_metal {
class IDevice;
class GlobalSemaphore;

namespace experimental {
// TODO: Move to impl when this class is PIMPLed
GlobalSemaphore CreateGlobalSemaphore(
    IDevice* device,
    const CoreRangeSet& cores,
    std::optional<uint32_t> initial_value,
    BufferType buffer_type,
    uint64_t address);
}  // namespace experimental
}  // namespace tt::tt_metal

namespace tt::tt_metal {

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

    static constexpr auto attribute_names = std::forward_as_tuple("cores", "buffer_type");
    auto attribute_values() const { return std::make_tuple(this->cores_, this->buffer_.get_buffer()->buffer_type()); }

private:
    // private constructor used by experimental API
    // TODO: Move to impl when this class is PIMPLed
    GlobalSemaphore(
        IDevice* device,
        const CoreRangeSet& cores,
        std::optional<uint32_t> initial_value,
        BufferType buffer_type,
        uint64_t address);

    void setup_buffer(
        std::optional<uint32_t> initial_value, BufferType buffer_type, std::optional<uint64_t> address = std::nullopt);

    // GlobalSemaphore is implemented as a wrapper around a sharded buffer
    // This can be updated in the future to be its own container with optimized dispatch functions
    distributed::AnyBuffer buffer_;
    IDevice* device_;
    CoreRangeSet cores_;

    friend GlobalSemaphore experimental::CreateGlobalSemaphore(
        IDevice* device,
        const CoreRangeSet& cores,
        std::optional<uint32_t> initial_value,
        BufferType buffer_type,
        uint64_t address);
};

}  // namespace tt::tt_metal

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::GlobalSemaphore& global_semaphore);

namespace std {

template <>
struct hash<tt::tt_metal::GlobalSemaphore> {
    std::size_t operator()(const tt::tt_metal::GlobalSemaphore& global_semaphore) const;
};

}  // namespace std
