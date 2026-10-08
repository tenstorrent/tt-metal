// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
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

// Where a global semaphore's address is reserved.
enum class GlobalSemaphorePlacement : uint8_t {
    // Lockstep: one address, kept clear on every core of the device.
    ALL_CORES,
    // Uniform per-core: one address, reserved only on the semaphore's own cores (HYBRID
    // allocator only; otherwise ALL_CORES). See per_core_allocation::set_uniform_address.
    OWN_CORES,
};

// GlobalSemaphoreImpl is implemented as a wrapper around a sharded buffer
// This can be updated in the future to be its own container with optimized dispatch functions
class GlobalSemaphoreImpl {
public:
    GlobalSemaphoreImpl(
        distributed::MeshDevice& device,
        CoreRangeSet cores,
        std::optional<uint32_t> initial_value,
        BufferType buffer_type,
        GlobalSemaphorePlacement placement = GlobalSemaphorePlacement::ALL_CORES);

    // Dedicated constructor for creating a global semaphore **without allocation**.
    // The instantiation of GlobalSemphore will be emplaced onto the address specified.
    GlobalSemaphoreImpl(
        distributed::MeshDevice& device,
        CoreRangeSet cores,
        std::optional<uint32_t> initial_value,
        BufferType buffer_type,
        uint64_t address);

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
        std::optional<uint32_t> initial_value,
        BufferType buffer_type,
        std::optional<uint64_t> address,
        GlobalSemaphorePlacement placement);

    std::shared_ptr<distributed::MeshBuffer> buffer_;
    distributed::MeshDevice* device_;
    CoreRangeSet cores_;
};

}  // namespace tt::tt_metal
