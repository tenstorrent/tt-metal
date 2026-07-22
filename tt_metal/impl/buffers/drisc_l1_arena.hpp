// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include <tt-metalium/hal_types.hpp>

#include "impl/context/context_types.hpp"

namespace tt::tt_metal {

class DriscL1Arena;

// RAII handle for a uniform-offset allocation inside the DRISC L1 "GCB zone".
// Held via shared_ptr by GlobalCircularBuffer so copies of the GCB share the
// same backing range. The destructor releases the slot back to the arena if
// the arena is still alive — if the owning MeshDevice has already torn down
// (and dropped the arena), the destructor is a no-op. Same lifetime pattern
// as MeshBuffer / MeshDevice.
class DriscL1Allocation {
public:
    DeviceAddr addr() const { return base_; }
    uint32_t size() const { return size_; }
    ~DriscL1Allocation();

    DriscL1Allocation(const DriscL1Allocation&) = delete;
    DriscL1Allocation& operator=(const DriscL1Allocation&) = delete;
    DriscL1Allocation(DriscL1Allocation&&) = delete;
    DriscL1Allocation& operator=(DriscL1Allocation&&) = delete;

private:
    friend class DriscL1Arena;
    DriscL1Allocation(std::weak_ptr<DriscL1Arena> arena, DeviceAddr base, uint32_t size) :
        arena_(std::move(arena)), base_(base), size_(size) {}

    std::weak_ptr<DriscL1Arena> arena_;
    DeviceAddr base_;
    uint32_t size_;
};

// Per-mesh, per-DRAM-bank arena for the DRISC L1 region above UNRESERVED.
//
// Layout:
//   [UNRESERVED, UNRESERVED + kGcbZoneSize)   — fixed zone for GCB pages_sent
//                                                 allocations (this arena).
//   [UNRESERVED + kGcbZoneSize, END)          — kernel working region for any
//                                                 long-lived DRISC kernel that
//                                                 co-exists with DRAM-sender
//                                                 GCBs (queried via
//                                                 kernel_working_region_base()).
//
// The zone is *fixed* so that allocating a GCB after such a kernel has started
// doesn't move the kernel's L1 layout: the kernel sits above the zone at a
// stable address.
// Lives as a std::shared_ptr on MeshDeviceImpl so that DriscL1Allocation handles
// can hold a weak_ptr back and survive close_impl() without UAF.
class DriscL1Arena : public std::enable_shared_from_this<DriscL1Arena> {
public:
    // Sized for ~16 GCBs at production receiver counts. Each GCB's per-bank
    // footprint is `2 * sizeof(uint32_t) * num_receivers_per_bank` bytes, e.g.
    // 2 * 4 * 8 = 64 B for ring=64 → 16 GCBs * 64 B = 1 KB exact (DRISC slots
    // are packed at 4-byte stride; the kernel walks them via
    // REMOTE_CB_LOCAL_PAGES_STRIDE under #ifdef COMPILE_FOR_DRISC). The
    // remaining ~92 KB above the zone is reported by kernel_working_region_size()
    // so callers placing a co-resident DRISC kernel know how much L1 they have.
    static constexpr uint32_t kGcbZoneSize = 1 * 1024;

    explicit DriscL1Arena(ContextId context_id);
    ~DriscL1Arena() = default;

    DriscL1Arena(const DriscL1Arena&) = delete;
    DriscL1Arena& operator=(const DriscL1Arena&) = delete;

    // Allocate `size` bytes inside the fixed GCB zone. The returned offset is the
    // same across all DRAM banks (uniform-offset constraint: every sender DRISC
    // core plants pages_sent at the same L1 offset). Similar in shape to the L1 /
    // DRAM bank allocators — a single pool, not per-bank.
    // TT_FATAL on invalid alignment; TT_THROW on zone full.
    std::shared_ptr<DriscL1Allocation> allocate(uint32_t size, uint32_t alignment);

    // Fixed base for the prefetcher kernel's working region. Unchanged for
    // the device's lifetime, regardless of current arena allocations.
    DeviceAddr kernel_working_region_base() const { return unreserved_base_ + kGcbZoneSize; }

    // Total DRISC L1 bytes available to the prefetcher kernel above the fixed
    // GCB zone. The manager uses this to size its ping-pong stage budget so
    // changing `kGcbZoneSize` automatically reduces the budget.
    uint32_t kernel_working_region_size() const { return drisc_unreserved_size_ - kGcbZoneSize; }

private:
    friend class DriscL1Allocation;
    void release(DeviceAddr base, uint32_t size);

    DeviceAddr unreserved_base_;
    uint32_t drisc_unreserved_size_;
    // Sorted ascending list of live (base, size) ranges within
    // [unreserved_base_, unreserved_base_ + kGcbZoneSize). First-fit allocate;
    // release coalesces with neighbors implicitly via removal.
    std::vector<std::pair<DeviceAddr, uint32_t>> live_;
    mutable std::mutex mutex_;
};

}  // namespace tt::tt_metal
