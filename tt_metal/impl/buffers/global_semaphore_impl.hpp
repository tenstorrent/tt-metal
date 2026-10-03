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

// GlobalSemaphoreImpl is implemented as a wrapper around a sharded buffer
// This can be updated in the future to be its own container with optimized dispatch functions
class GlobalSemaphoreImpl {
public:
    // `cq_id` is the queue for the initial-value write with fast dispatch (cq 0 if not provided).
    GlobalSemaphoreImpl(
        distributed::MeshDevice& device,
        CoreRangeSet cores,
        std::optional<uint32_t> initial_value,
        BufferType buffer_type,
        std::optional<uint8_t> cq_id = std::nullopt);

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

    // With fast dispatch, resets via command queue `cq_id` (cq 0 if not provided). The queue is not looked up on
    // slow dispatch / simulator.
    void reset_semaphore_value(uint32_t reset_value, std::optional<uint8_t> cq_id = std::nullopt) const;

private:
    void setup_buffer(
        std::optional<uint32_t> initial_value,
        BufferType buffer_type,
        std::optional<uint64_t> address,
        std::optional<uint8_t> cq_id = std::nullopt);

    std::shared_ptr<distributed::MeshBuffer> buffer_;
    distributed::MeshDevice* device_;
    CoreRangeSet cores_;
};

}  // namespace tt::tt_metal
