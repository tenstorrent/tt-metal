// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_event.hpp>

namespace tt::tt_metal {

class AllocatorImpl;
namespace distributed {
class MeshBuffer;
class MeshCommandQueue;
class MeshDevice;
class MeshDeviceImpl;
}  // namespace distributed

// The pool's block of L1 on every worker core. Each submesh marks the same range as used in its own allocator, so
// that allocator does not place anything there.
class GlobalSemaphorePoolReservation {
public:
    // Takes `size` bytes from the L1 allocator of `device`.
    static std::unique_ptr<GlobalSemaphorePoolReservation> from_allocator(
        distributed::MeshDevice& device, const CoreRangeSet& worker_grid, DeviceAddr size);
    // Marks the parent's allocator block [address, address + size) as used in a submesh's allocator.
    static std::unique_ptr<GlobalSemaphorePoolReservation> mirror_allocator_block(
        AllocatorImpl& allocator, DeviceAddr address, DeviceAddr size);

    ~GlobalSemaphorePoolReservation();

    GlobalSemaphorePoolReservation(const GlobalSemaphorePoolReservation&) = delete;
    GlobalSemaphorePoolReservation& operator=(const GlobalSemaphorePoolReservation&) = delete;

    DeviceAddr address() const { return address_; }

private:
    GlobalSemaphorePoolReservation() = default;

    AllocatorImpl* allocator_ = nullptr;
    std::shared_ptr<distributed::MeshBuffer> buffer_;
    bool mirrored_ = false;
    DeviceAddr address_ = 0;
};

class GlobalSemaphorePool;

// Shared by all copies of a pool GlobalSemaphore. Returns the slot when the last copy is destroyed.
struct GlobalSemaphoreSlotLease {
    std::weak_ptr<GlobalSemaphorePool> pool;
    std::weak_ptr<distributed::MeshDevice> device;
    CoreRangeSet cores;
    DeviceAddr address = 0;
    uint32_t slot = 0;

    GlobalSemaphoreSlotLease(
        std::weak_ptr<GlobalSemaphorePool> pool,
        std::weak_ptr<distributed::MeshDevice> device,
        CoreRangeSet cores,
        DeviceAddr address,
        uint32_t slot);
    ~GlobalSemaphoreSlotLease();

    GlobalSemaphoreSlotLease(const GlobalSemaphoreSlotLease&) = delete;
    GlobalSemaphoreSlotLease& operator=(const GlobalSemaphoreSlotLease&) = delete;
};

// Fixed-size slots with the same address on every core and chip of the mesh. A returned slot is not reused
// until the work already queued on that mesh's command queues has finished.
class GlobalSemaphorePool : public std::enable_shared_from_this<GlobalSemaphorePool> {
public:
    GlobalSemaphorePool(
        std::unique_ptr<GlobalSemaphorePoolReservation> reservation,
        CoreRangeSet worker_grid,
        uint32_t num_slots,
        DeviceAddr slot_size);

    GlobalSemaphorePool(const GlobalSemaphorePool&) = delete;
    GlobalSemaphorePool& operator=(const GlobalSemaphorePool&) = delete;

    static DeviceAddr reservation_size(uint32_t num_slots, DeviceAddr slot_size) {
        return static_cast<DeviceAddr>(num_slots) * slot_size;
    }

    // Called when the owning MeshDevice closes. Frees the L1; acquire() throws after this.
    void close();

    const CoreRangeSet& worker_grid() const { return worker_grid_; }
    uint32_t num_slots() const { return num_slots_; }
    DeviceAddr base_address() const { return base_address_; }
    DeviceAddr slot_size() const { return slot_size_; }
    uint32_t num_live() const;
    uint32_t num_pending() const;
    uint32_t peak_live() const;

    // If no slot is free, waits for a returned slot to become reusable.
    std::shared_ptr<const GlobalSemaphoreSlotLease> acquire(distributed::MeshCommandQueue& cq, const CoreRangeSet& cores);
    void release(uint32_t slot, std::weak_ptr<distributed::MeshDevice> device) noexcept;

    // Called from quiesce_devices(), which waits for all work and resets the event counters. Frees the slots
    // of `device` that were waiting on events from before the reset.
    void device_quiesced(const distributed::MeshDeviceImpl& device);

private:
    struct PendingSlot {
        uint32_t slot = 0;
        std::weak_ptr<distributed::MeshDevice> device;
        std::optional<std::vector<distributed::MeshEvent>> events;  // Recorded on the next acquire.
    };

    void record_events_for_pending();
    void collect_completed_pending();  // Caller holds mutex_.

    std::unique_ptr<GlobalSemaphorePoolReservation> reservation_;  // Guarded by mutex_.
    DeviceAddr base_address_;
    CoreRangeSet worker_grid_;
    uint32_t num_slots_;
    DeviceAddr slot_size_;

    mutable std::mutex mutex_;
    std::vector<bool> slot_in_use_;  // Live or pending.
    std::vector<PendingSlot> pending_;
    uint32_t num_live_ = 0;
    uint32_t peak_live_ = 0;
};

// Writes `value` to `address` on `cores` of every local chip of `cq`'s device and waits until it is done. In fast
// dispatch the write goes through `cq`, after the programs already enqueued on it.
void write_global_semaphore_value(
    distributed::MeshCommandQueue& cq, const CoreRangeSet& cores, DeviceAddr address, uint32_t value);

}  // namespace tt::tt_metal
