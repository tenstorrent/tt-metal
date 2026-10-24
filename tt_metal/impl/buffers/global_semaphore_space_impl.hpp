// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/global_semaphore_space.hpp>

namespace tt::tt_metal::experimental {

struct GlobalSemaphoreLease;

// One PersistentL1Arena allocation of capacity * slot_stride bytes over the space's cores, taken at
// creation. Semaphores lease slots from it without touching the arena again, so creating one works
// after programs have sealed those cores. Every live semaphore owns a distinct slot, which keeps the
// address the same on all of its cores.
class GlobalSemaphoreSpaceImpl : public std::enable_shared_from_this<GlobalSemaphoreSpaceImpl> {
public:
    GlobalSemaphoreSpaceImpl(distributed::MeshDevice& device, CoreRangeSet cores, uint32_t capacity);
    ~GlobalSemaphoreSpaceImpl();

    GlobalSemaphoreSpaceImpl(const GlobalSemaphoreSpaceImpl&) = delete;
    GlobalSemaphoreSpaceImpl& operator=(const GlobalSemaphoreSpaceImpl&) = delete;

    distributed::MeshDevice& device() const { return *device_; }
    const CoreRangeSet& cores() const { return cores_; }
    uint32_t capacity() const { return capacity_; }
    uint32_t num_live_semaphores() const;
    DeviceAddr base_address() const { return base_address_; }
    DeviceAddr slot_stride() const { return slot_stride_; }

    std::shared_ptr<const GlobalSemaphoreLease> acquire(const CoreRangeSet& cores);
    void release(uint32_t slot) noexcept;

private:
    distributed::MeshDevice* device_;
    // Semaphores may outlive the MeshDevice; the destructor only frees the region if it is still alive.
    std::weak_ptr<distributed::MeshDevice> weak_device_;
    CoreRangeSet cores_;
    uint32_t capacity_;
    DeviceAddr slot_stride_ = 0;
    DeviceAddr base_address_ = 0;
    uint64_t allocation_id_ = 0;

    mutable std::mutex mutex_;
    std::vector<bool> slot_in_use_;
    uint32_t num_live_ = 0;
};

// Shared by every copy of one GlobalSemaphore created from a space; returns the slot when the last
// copy goes away.
struct GlobalSemaphoreLease {
    std::shared_ptr<GlobalSemaphoreSpaceImpl> space;
    CoreRangeSet cores;
    DeviceAddr address = 0;
    uint32_t slot = 0;

    GlobalSemaphoreLease(
        std::shared_ptr<GlobalSemaphoreSpaceImpl> space, CoreRangeSet cores, DeviceAddr address, uint32_t slot);
    ~GlobalSemaphoreLease();

    GlobalSemaphoreLease(const GlobalSemaphoreLease&) = delete;
    GlobalSemaphoreLease& operator=(const GlobalSemaphoreLease&) = delete;
};

// Writes `value` to `address` on every core of `cores`, on every local device of `device`, and waits
// for it to land. Fast dispatch orders it after previously enqueued programs on command queue 0.
void write_global_semaphore_value(
    distributed::MeshDevice& device, const CoreRangeSet& cores, DeviceAddr address, uint32_t value);

}  // namespace tt::tt_metal::experimental
