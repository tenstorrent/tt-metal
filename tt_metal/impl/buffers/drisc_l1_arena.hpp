// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>

#include "impl/context/context_types.hpp"

namespace tt::tt_metal {

class DriscL1Arena;

// RAII handle for an allocation inside the DRISC L1 reserved zone. Held via
// shared_ptr by its owner (a DRAM-sender PrefetcherPipe) so copies share the
// same backing range. The destructor
// releases the range back to the arena if the arena is still alive — if the
// owning MeshDevice has already torn down (and dropped the arena), the
// destructor is a no-op. Same lifetime pattern as MeshBuffer / MeshDevice.
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
    DriscL1Allocation(std::weak_ptr<DriscL1Arena> arena, DeviceAddr base, uint32_t size, const CoreCoord& core) :
        arena_(std::move(arena)), base_(base), size_(size), core_(core) {}

    std::weak_ptr<DriscL1Arena> arena_;
    DeviceAddr base_;
    uint32_t size_;
    CoreCoord core_;
};

// Per-mesh, per-DRAM-bank arena for the DRISC L1 region above UNRESERVED.
//
// Layout:
//   [UNRESERVED, UNRESERVED + kSenderStateZoneSize)   — fixed zone this arena hands out
//                                                 to DRAM-sender state
//                                                 (PrefetcherPipe sender config
//                                                 pages).
//   [UNRESERVED + kSenderStateZoneSize, END)          — kernel working region for any
//                                                 long-lived DRISC kernel that
//                                                 co-exists with those senders
//                                                 (queried via
//                                                 kernel_working_region_base()).
//
// The zone is *fixed* so that allocating a sender after such a kernel has
// started doesn't move the kernel's L1 layout: the kernel sits above the zone at
// a stable address.
//
// A range is reserved on a single named DRAM sender core (`allocate_on`), so two
// ranges collide only when they are on the same core.
// Lives as a std::shared_ptr on MeshDeviceImpl so that DriscL1Allocation handles
// can hold a weak_ptr back and survive close_impl() without UAF.
class DriscL1Arena : public std::enable_shared_from_this<DriscL1Arena> {
public:
    // A PrefetcherPipe set's per-core config pages all share one offset, so a
    // whole set costs one page per sender core: 384 B at 8 receivers (a 40 B
    // header and 8 NOC XY pairs padded to 128 B, then the SENT and ACKED blocks
    // of 8 L1-aligned slots, 128 B each on their own 64 B lines), so the zone
    // holds two such sets live on a core at once. The remaining ~92 KB above the
    // zone is reported by kernel_working_region_size() so callers placing a
    // co-resident DRISC kernel know how much L1 they have.
    static constexpr uint32_t kSenderStateZoneSize = 1 * 1024;

    explicit DriscL1Arena(ContextId context_id);
    ~DriscL1Arena() = default;

    DriscL1Arena(const DriscL1Arena&) = delete;
    DriscL1Arena& operator=(const DriscL1Arena&) = delete;

    // Allocate `size` bytes on the DRAM sender core `core` (logical) alone,
    // leaving the same address free on every other bank. A sender is handed its
    // state address explicitly (a DRAM-sender PrefetcherPipe passes it in the
    // request header), so N single-sender objects cost one range each rather
    // than N zone-wide offsets.
    // TT_FATAL on invalid alignment; TT_THROW on zone full.
    std::shared_ptr<DriscL1Allocation> allocate_on(const CoreCoord& core, uint32_t size, uint32_t alignment);

    // Fixed base for the prefetcher kernel's working region. Unchanged for
    // the device's lifetime, regardless of current arena allocations.
    DeviceAddr kernel_working_region_base() const { return unreserved_base_ + kSenderStateZoneSize; }

    // Total DRISC L1 bytes available to the prefetcher kernel above the fixed
    // sender-state zone. The manager uses this to size its ping-pong stage budget so
    // changing `kSenderStateZoneSize` automatically reduces the budget.
    uint32_t kernel_working_region_size() const { return drisc_unreserved_size_ - kSenderStateZoneSize; }

private:
    friend class DriscL1Allocation;
    void release(DeviceAddr base, uint32_t size, const CoreCoord& core);

    struct LiveRange {
        DeviceAddr base;
        uint32_t size;
        // The one DRAM sender core it occupies.
        CoreCoord core;
    };

    DeviceAddr unreserved_base_;
    uint32_t drisc_unreserved_size_;
    // Live ranges within [unreserved_base_, unreserved_base_ + kSenderStateZoneSize), sorted
    // ascending by base. First-fit allocate over the ranges on the requested core; release
    // coalesces with neighbors implicitly via removal.
    std::vector<LiveRange> live_;
    mutable std::mutex mutex_;
};

}  // namespace tt::tt_metal
