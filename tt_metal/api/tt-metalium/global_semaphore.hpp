// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
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

// forward declarations
namespace tt::tt_metal {
class IDevice;
class GlobalSemaphore;

namespace experimental {
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

    GlobalSemaphore(const GlobalSemaphore&);
    GlobalSemaphore& operator=(const GlobalSemaphore&);

    GlobalSemaphore(GlobalSemaphore&&) noexcept;
    GlobalSemaphore& operator=(GlobalSemaphore&&) noexcept;

    ~GlobalSemaphore();

    IDevice* device() const;

    DeviceAddr address() const;

    void reset_semaphore_value(uint32_t reset_value) const;

    static constexpr auto attribute_names = std::forward_as_tuple("cores", "buffer_type");
    std::tuple<CoreRangeSet, BufferType> attribute_values() const;

private:
    // private constructor used by experimental API
    GlobalSemaphore(
        IDevice* device,
        const CoreRangeSet& cores,
        std::optional<uint32_t> initial_value,
        BufferType buffer_type,
        uint64_t address);

    struct Impl;
    std::unique_ptr<Impl> pimpl_;

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
