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
// shared_ptr by its owner (a GlobalCircularBuffer or a DRAM-sender
// PrefetcherPipe) so copies share the same backing range. The destructor
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
        std::weak_ptr<DriscL1Arena> arena, DeviceAddr base, uint32_t size, std::optional<CoreCoord> core) :
        arena_(std::move(arena)), base_(base), size_(size), core_(core) {}

    std::weak_ptr<DriscL1Arena> arena_;
    DeviceAddr base_;
    uint32_t size_;
    std::optional<CoreCoord> core_;
};

// Per-mesh, per-DRAM-bank arena for the DRISC L1 region above UNRESERVED.
//
// Layout:
//   [UNRESERVED, + sender_state_zone_size())          — fixed zone this arena hands out
//                                                 to DRAM-sender remote-buffer
//                                                 state (GCB pages_sent + state
//                                                 blocks, PrefetcherPipe sender
//                                                 config pages).
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
// receiver zone; Quasar has kQuasarSenderStateZoneSize and kQuasarReceiverZoneSize.
//
// A range is reserved either on every bank (`allocate`, for senders that must
// sit at one uniform offset) or on a single named DRAM sender core
// (`allocate_on`). Two ranges may share addresses as long as no bank sees both:
// a per-core range collides only with a uniform range or with another range on
// the same core.
// Lives as a std::shared_ptr on MeshDeviceImpl so that DriscL1Allocation handles
// can hold a weak_ptr back and survive close_impl() without UAF.
class DriscL1Arena : public std::enable_shared_from_this<DriscL1Arena> {
public:
    // Blackhole: sized for ~16 GCBs at production receiver counts. Each GCB's per-bank
    // footprint is `2 * sizeof(uint32_t) * num_receivers_per_bank` bytes, e.g.
    // 2 * 4 * 8 = 64 B for ring=64 → 16 GCBs * 64 B = 1 KB exact (DRISC slots
    // are packed at 4-byte stride; the kernel walks them via
    // REMOTE_CB_LOCAL_PAGES_STRIDE under #ifdef COMPILE_FOR_DRISC). A
    // PrefetcherPipe set is coarser but rarer: its per-core config pages all
    // share one offset, so a whole set costs one page (~356 B for 8 receivers:
    // 36 B header + 8 NOC XY pairs + 8 L1-aligned counter pairs). The remaining
    // ~92 KB above the zone is reported by kernel_working_region_size() so
    // callers placing a co-resident DRISC kernel know how much L1 they have.
    //
    // Quasar: kQuasarSenderStateZoneSize and kQuasarReceiverZoneSize. A GCB with DRAM receivers
    // gives each pages_sent and pages_acked counter its own line (dram_receiver_config_page.hpp),
    // and the receiver zone holds those GCBs' receiver config pages and FIFOs.
    static constexpr uint32_t kSenderStateZoneSize = 1 * 1024;
    static constexpr uint32_t kQuasarSenderStateZoneSize = 16 * 1024;
    static constexpr uint32_t kQuasarReceiverZoneSize = 1024 * 1024;

    explicit DriscL1Arena(ContextId context_id);
    ~DriscL1Arena() = default;

    DriscL1Arena(const DriscL1Arena&) = delete;
    DriscL1Arena& operator=(const DriscL1Arena&) = delete;

    // Allocate `size` bytes at one offset reserved on *every* DRAM bank, for a
    // sender whose kernel finds its state at a uniform address (the GCB
    // pages_sent / state block pair). Similar in shape to the L1 / DRAM bank
    // allocators — a single pool, not per-bank.
    // TT_FATAL on invalid alignment; TT_THROW on zone full.
    std::shared_ptr<DriscL1Allocation> allocate(uint32_t size, uint32_t alignment);

    // Allocate `size` bytes on `dram_sender_logical` alone, leaving the same
    // address free on every other bank. Senders that are handed their state
    // address explicitly (a DRAM-sender PrefetcherPipe passes it in the request
    // header) take this form, so N single-sender objects cost one range each
    // rather than N zone-wide offsets.
    // TT_FATAL on invalid alignment; TT_THROW on zone full.
    std::shared_ptr<DriscL1Allocation> allocate_on(
        const CoreCoord& dram_sender_logical, uint32_t size, uint32_t alignment);

    // Allocate `size` bytes at one offset in the receiver zone, reserved on every DRAM core, for a
    // GCB whose receivers are DRAM cores. Not keyed by logical core: two logical coords can name one
    // core. TT_FATAL when the arch has no receiver zone or on invalid alignment; TT_THROW on zone full.
    std::shared_ptr<DriscL1Allocation> allocate_receiver(uint32_t size, uint32_t alignment);

    uint32_t sender_state_zone_size() const { return sender_state_zone_size_; }
    DeviceAddr receiver_zone_base() const { return unreserved_base_ + sender_state_zone_size_; }
    uint32_t receiver_zone_size() const { return receiver_zone_size_; }

    // Fixed base for the prefetcher kernel's working region. Unchanged for
    // the device's lifetime, regardless of current arena allocations.
    DeviceAddr kernel_working_region_base() const { return receiver_zone_base() + receiver_zone_size_; }

    // Total DRISC L1 bytes available to the prefetcher kernel above the fixed
    // zones. The manager uses this to size its ping-pong stage budget so
    // growing a zone automatically reduces the budget.
    uint32_t kernel_working_region_size() const {
        return drisc_unreserved_size_ - sender_state_zone_size_ - receiver_zone_size_;
    }

private:
    friend class DriscL1Allocation;
    std::shared_ptr<DriscL1Allocation> allocate_impl(
        std::optional<CoreCoord> core, DeviceAddr zone_begin, uint32_t zone_size, uint32_t size, uint32_t alignment);
    void release(DeviceAddr base, uint32_t size, const std::optional<CoreCoord>& core);

    struct LiveRange {
        DeviceAddr base;
        uint32_t size;
        // nullopt = reserved on every bank; otherwise the one DRAM sender core it
        // occupies.
        std::optional<CoreCoord> core;
    };

    DeviceAddr unreserved_base_;
    uint32_t drisc_unreserved_size_;
    uint32_t sender_state_zone_size_;
    uint32_t receiver_zone_size_;
    // Live ranges within both zones, sorted ascending by base. First-fit allocate over the
    // ranges that share a bank with the request; release coalesces with neighbors implicitly
    // via removal.
    std::vector<LiveRange> live_;
    mutable std::mutex mutex_;
};

}  // namespace tt::tt_metal
