// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
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
    DriscL1Allocation(
        std::weak_ptr<DriscL1Arena> arena, DeviceAddr base, uint32_t size, const std::optional<CoreCoord>& core) :
        arena_(std::move(arena)), base_(base), size_(size), core_(core) {}

    std::weak_ptr<DriscL1Arena> arena_;
    DeviceAddr base_;
    uint32_t size_;
    // The one core it occupies; unset when it is reserved on every core.
    std::optional<CoreCoord> core_;
};

// Per-mesh, per-DRAM-bank arena for the DRISC L1 region above UNRESERVED.
//
// Layout:
//   [UNRESERVED, + sender_state_zone_size())          — fixed zone this arena hands out
//                                                 to DRAM-sender state
//                                                 (PrefetcherPipe sender config
//                                                 pages, and the sender state of
//                                                 GCBs whose receivers are DRAM
//                                                 cores).
//   [receiver_zone_base(), + receiver_zone_size())    — Quasar only: receiver config
//                                                 pages and FIFOs of GCBs whose
//                                                 receivers are DRAM cores.
//   [kernel_working_region_base(), END)               — kernel working region for any
//                                                 long-lived DRISC kernel that
//                                                 co-exists with those senders.
//
// The zones are *fixed* so that allocating a sender after such a kernel has
// started doesn't move the kernel's L1 layout: the kernel sits above them at
// a stable address. Blackhole has a kSenderStateZoneSize sender-state zone and no
// receiver zone; CCE DRAM cores have kCceSenderStateZoneSize and kCceReceiverZoneSize.
//
// A range is reserved either on a single named DRAM sender core (`allocate_on`), so
// two such ranges collide only when they are on the same core, or on every core
// (`allocate`, `allocate_receiver`, for state that must sit at one offset on all of
// them), which collides with every range in its zone.
//
// Where DRAM-sender endpoints run on the dispatch engine instead (for_dispatch_engine),
// the zone is a fixed slice of that engine's L1 and every endpoint shares it: ranges
// collide whichever endpoint they were reserved for, and there is no working region
// (the dispatch memory map lays out the rest of that L1).
// Lives as a std::shared_ptr on MeshDeviceImpl so that DriscL1Allocation handles
// can hold a weak_ptr back and survive close_impl() without UAF.
class DriscL1Arena : public std::enable_shared_from_this<DriscL1Arena> {
public:
    // Blackhole: a PrefetcherPipe set's per-core config pages all share one offset, so a
    // whole set costs one page per sender core: 384 B at 8 receivers (a 40 B
    // header and 8 NOC XY pairs padded to 128 B, then the SENT and ACKED blocks
    // of 8 L1-aligned slots, 128 B each on their own 64 B lines), so the zone
    // holds two such sets live on a core at once. The remaining ~92 KB above the
    // zone is reported by kernel_working_region_size() so callers placing a
    // co-resident DRISC kernel know how much L1 they have.
    //
    // CCE DRAM cores: kCceSenderStateZoneSize and kCceReceiverZoneSize. A pipe set with DRAM receivers
    // gives each pages_sent and pages_acked counter its own line (dram_receiver_config_page.hpp),
    // and the receiver zone holds those sets' receiver config pages and FIFOs.
    static constexpr uint32_t kSenderStateZoneSize = 1 * 1024;
    static constexpr uint32_t kCceSenderStateZoneSize = 16 * 1024;
    static constexpr uint32_t kCceReceiverZoneSize = 1024 * 1024;

    explicit DriscL1Arena(ContextId context_id);
    ~DriscL1Arena() = default;

    // Sender-state zone [zone_base, zone_base + zone_size) of the dispatch engine that hosts every
    // DRAM-sender endpoint (DramSenderPlacement::DispatchEngine).
    static std::shared_ptr<DriscL1Arena> for_dispatch_engine(DeviceAddr zone_base, uint32_t zone_size);

    DriscL1Arena(const DriscL1Arena&) = delete;
    DriscL1Arena& operator=(const DriscL1Arena&) = delete;

    // Allocate `size` bytes on the DRAM sender core `core` (logical) alone,
    // leaving the same address free on every other bank. A sender is handed its
    // state address explicitly (a DRAM-sender PrefetcherPipe passes it in the
    // request header), so N single-sender objects cost one range each rather
    // than N zone-wide offsets.
    // TT_FATAL on invalid alignment; TT_THROW on zone full.
    std::shared_ptr<DriscL1Allocation> allocate_on(const CoreCoord& core, uint32_t size, uint32_t alignment);

    // Allocate `size` bytes at one offset in the sender-state zone, reserved on every DRAM core, for
    // sender state that every bank's sender finds at the same address.
    // TT_FATAL on invalid alignment; TT_THROW on zone full.
    std::shared_ptr<DriscL1Allocation> allocate(uint32_t size, uint32_t alignment);

    // Allocate `size` bytes at one offset in the receiver zone, reserved on every DRAM core, for a
    // GCB whose receivers are DRAM cores. Not keyed by logical core: two logical coords can name one
    // core. TT_FATAL when the arch has no receiver zone or on invalid alignment; TT_THROW on zone full.
    std::shared_ptr<DriscL1Allocation> allocate_receiver(uint32_t size, uint32_t alignment);

    uint32_t sender_state_zone_size() const { return sender_state_zone_size_; }
    DeviceAddr receiver_zone_base() const { return unreserved_base_ + sender_state_zone_size_; }
    uint32_t receiver_zone_size() const { return receiver_zone_size_; }

    // Fixed base for the prefetcher kernel's working region. Unchanged for
    // the device's lifetime, regardless of current arena allocations. DRAM cores only.
    DeviceAddr kernel_working_region_base() const { return receiver_zone_base() + receiver_zone_size_; }

    // Total DRISC L1 bytes available to the prefetcher kernel above the fixed
    // zones. The manager uses this to size its ping-pong stage budget so
    // growing a zone automatically reduces the budget. DRAM cores only.
    uint32_t kernel_working_region_size() const {
        return drisc_unreserved_size_ - sender_state_zone_size_ - receiver_zone_size_;
    }

private:
    // Whether every endpoint's ranges live in one shared L1.
    enum class EndpointL1 : uint8_t { PerEndpoint, Shared };

    // A zone every endpoint shares (for_dispatch_engine).
    DriscL1Arena(DeviceAddr zone_base, uint32_t zone_size);

    friend class DriscL1Allocation;
    std::shared_ptr<DriscL1Allocation> allocate_impl(
        std::optional<CoreCoord> core, DeviceAddr zone_begin, uint32_t zone_size, uint32_t size, uint32_t alignment);
    void release(DeviceAddr base, uint32_t size, const std::optional<CoreCoord>& core);
    // A range reserved on every core (no core named) shares an L1 with every other range.
    bool same_l1(const std::optional<CoreCoord>& a, const std::optional<CoreCoord>& b) const {
        return endpoint_l1_ == EndpointL1::Shared || !a.has_value() || !b.has_value() || *a == *b;
    }

    struct LiveRange {
        DeviceAddr base;
        uint32_t size;
        // The one DRAM sender core it occupies; unset when it is reserved on every core.
        std::optional<CoreCoord> core;
    };

    DeviceAddr unreserved_base_;
    uint32_t drisc_unreserved_size_;
    uint32_t sender_state_zone_size_ = kSenderStateZoneSize;
    uint32_t receiver_zone_size_ = 0;
    EndpointL1 endpoint_l1_ = EndpointL1::PerEndpoint;
    // Live ranges within both zones, sorted ascending by base. First-fit allocate over the
    // ranges that share an L1 with the request; release coalesces with neighbors implicitly
    // via removal.
    std::vector<LiveRange> live_;
    mutable std::mutex mutex_;
};

}  // namespace tt::tt_metal
