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

// Selects the PersistentL1Arena-backed constructor.
struct PersistentL1Placement {};

// GlobalSemaphoreImpl is backed by either a sharded L1 buffer (the default) or a 4-byte-per-core
// PersistentL1Arena allocation (opt-in). Both give the semaphore the same address on every core and device.
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

    // Allocates the semaphore in the PersistentL1Arena. Must run before any program places
    // program-local L1 on these cores.
    GlobalSemaphoreImpl(
        distributed::MeshDevice& device,
        CoreRangeSet cores,
        std::optional<uint32_t> initial_value,
        PersistentL1Placement placement);

    // Copy/move semantics
    GlobalSemaphoreImpl(const GlobalSemaphoreImpl&) = default;
    GlobalSemaphoreImpl& operator=(const GlobalSemaphoreImpl&) = default;
    GlobalSemaphoreImpl(GlobalSemaphoreImpl&&) noexcept = default;
    GlobalSemaphoreImpl& operator=(GlobalSemaphoreImpl&&) noexcept = default;

    distributed::MeshDevice& device() const;

    const CoreRangeSet& cores() const;

    BufferType buffer_type() const;

    DeviceAddr address() const;

    bool is_persistent() const;

    void reset_semaphore_value(uint32_t reset_value) const;

private:
    // Frees its arena allocation when the last GlobalSemaphoreImpl copy sharing it is destroyed.
    struct PersistentAllocation {
        distributed::MeshDevice* device = nullptr;
        uint64_t id = 0;
        DeviceAddr address = 0;
        ~PersistentAllocation();
    };

    void setup_buffer(
        std::optional<uint32_t> initial_value, BufferType buffer_type, std::optional<uint64_t> address);
    void setup_persistent(std::optional<uint32_t> initial_value);
    void reset_persistent_value(uint32_t reset_value) const;

    std::shared_ptr<distributed::MeshBuffer> buffer_;
    std::shared_ptr<PersistentAllocation> persistent_;
    distributed::MeshDevice* device_;
    CoreRangeSet cores_;
};

}  // namespace tt::tt_metal
