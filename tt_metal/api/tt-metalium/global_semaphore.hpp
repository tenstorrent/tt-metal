// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>

// forward declarations
namespace tt::tt_metal {
class IDevice;
class GlobalSemaphore;
class GlobalSemaphoreImpl;

GlobalSemaphore CreateGlobalSemaphore(
    IDevice* device, const CoreRangeSet& cores, uint32_t initial_value, BufferType buffer_type);
GlobalSemaphore CreateGlobalSemaphore(
    IDevice* device, CoreRangeSet&& cores, uint32_t initial_value, BufferType buffer_type);

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
    GlobalSemaphore(const GlobalSemaphore& other);
    GlobalSemaphore& operator=(const GlobalSemaphore& other);

    GlobalSemaphore(GlobalSemaphore&& other) noexcept;
    GlobalSemaphore& operator=(GlobalSemaphore&& other) noexcept;

    ~GlobalSemaphore();

    IDevice* device() const;

    DeviceAddr address() const;

    void reset_semaphore_value(uint32_t reset_value) const;

    static constexpr auto attribute_names = std::forward_as_tuple("cores", "buffer_type");
    std::tuple<CoreRangeSet, BufferType> attribute_values() const;

    GlobalSemaphoreImpl& impl();
    const GlobalSemaphoreImpl& impl() const;

private:
    GlobalSemaphore(
        IDevice* device, const CoreRangeSet& cores, uint32_t initial_value, BufferType buffer_type = BufferType::L1);

    GlobalSemaphore(
        IDevice* device, CoreRangeSet&& cores, uint32_t initial_value, BufferType buffer_type = BufferType::L1);

    // Internal constructor (internal use only)
    GlobalSemaphore(GlobalSemaphoreImpl&& impl);

    std::unique_ptr<GlobalSemaphoreImpl> pimpl_;

    friend GlobalSemaphore CreateGlobalSemaphore(
        IDevice* device, const CoreRangeSet& cores, uint32_t initial_value, BufferType buffer_type);
    friend GlobalSemaphore CreateGlobalSemaphore(
        IDevice* device, CoreRangeSet&& cores, uint32_t initial_value, BufferType buffer_type);
    friend GlobalSemaphore experimental::CreateGlobalSemaphore(
        IDevice* device,
        const CoreRangeSet& cores,
        std::optional<uint32_t> initial_value,
        BufferType buffer_type,
        uint64_t address);
};

}  // namespace tt::tt_metal
