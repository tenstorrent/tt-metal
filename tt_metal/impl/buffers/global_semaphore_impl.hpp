// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>

namespace tt::tt_metal {

class GlobalSemaphore;
namespace distributed {
class MeshDevice;
}  // namespace distributed
namespace experimental {
struct GlobalSemaphoreLease;
}  // namespace experimental

// GlobalSemaphoreImpl is implemented as a wrapper around a sharded buffer, or around a slot leased
// from an experimental::GlobalSemaphoreSpace.
// This can be updated in the future to be its own container with optimized dispatch functions
class GlobalSemaphoreImpl {
public:
    GlobalSemaphoreImpl(
        distributed::MeshDevice& device,
        CoreRangeSet cores,
        std::optional<uint32_t> initial_value,
        BufferType buffer_type);

    // Dedicated constructor for creating a global semaphore **without allocation**.
    // The instantiation of GlobalSemphore will be emplaced onto the address specified.
    GlobalSemaphoreImpl(
        distributed::MeshDevice& device,
        CoreRangeSet cores,
        std::optional<uint32_t> initial_value,
        BufferType buffer_type,
        uint64_t address);

    // Backed by a slot of an experimental::GlobalSemaphoreSpace; copies share the slot.
    explicit GlobalSemaphoreImpl(std::shared_ptr<const experimental::GlobalSemaphoreLease> lease);

    // Copy/move semantics
    GlobalSemaphoreImpl(const GlobalSemaphoreImpl&) = default;
    GlobalSemaphoreImpl& operator=(const GlobalSemaphoreImpl&) = default;
    GlobalSemaphoreImpl(GlobalSemaphoreImpl&&) noexcept = default;
    GlobalSemaphoreImpl& operator=(GlobalSemaphoreImpl&&) noexcept = default;

    distributed::MeshDevice& device() const;

    const CoreRangeSet& cores() const;

    BufferType buffer_type() const;

    DeviceAddr address() const;

    void reset_semaphore_value(uint32_t reset_value) const;

private:
    void setup_buffer(
        std::optional<uint32_t> initial_value, BufferType buffer_type, std::optional<uint64_t> address);

    std::shared_ptr<distributed::MeshBuffer> buffer_;
    std::shared_ptr<const experimental::GlobalSemaphoreLease> lease_;
    distributed::MeshDevice* device_;
    CoreRangeSet cores_;
};

}  // namespace tt::tt_metal
