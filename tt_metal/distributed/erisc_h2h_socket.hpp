// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The bridge's host-to-host hop. H2HSocket's shape, but moving [packet | BridgeDescriptor] slots
// with arena geometry, not socket pages. Two hosts means two clocks: round trip only, no one-way.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "tt_metal/distributed/erisc_bridge_tasks.hpp"

namespace tt::tt_metal::experimental {

class EriscH2HSocket {
public:
    struct Config {
        // H2HSocket counts cores; the bridge counts arenas -- (link, receiver channel) pairs.
        uint32_t arenas = 0;
        uint32_t chans_per_link = 1;
        uint32_t page_bytes = 0;  // bridge_slot_size(packet_capacity)
        uint32_t ring_pages = 1;
        uint32_t peer_rank = 1;
        // This host's id and the host count. They are handed to MPI as ranks, so create() refuses
        // a mismatch with MPI_COMM_WORLD rather than landing every frame on the wrong peer.
        uint32_t host_rank = 0;
        uint32_t host_count = 2;
        // The pinned region both halves address by offset, so no address is ever exchanged.
        uint8_t* region_base = nullptr;
        uint64_t region_bytes = 0;
        uint32_t max_queued_frames = 0;  // frames queued but not yet put; 0 = unbounded
        uint32_t max_put_bytes = 0;      // cap on a coalesced payload put (MPI's eager limit); 0 = none
        uint32_t max_batch = 0;          // frames per poll(); 0 = half of arenas * ring, so put and drain overlap
        bool collect_timing = false;
        uint64_t timing_samples = 0;  // RTT samples reserved up front: growing them mid-run stalls for ms
    };

    // Collective: every rank must call it, including one that already failed locally, or the
    // ranks that succeeded hang in the next collective.
    static std::unique_ptr<EriscH2HSocket> create(const Config& cfg, std::string& err);
    ~EriscH2HSocket();

    EriscH2HSocket(const EriscH2HSocket&) = delete;
    EriscH2HSocket& operator=(const EriscH2HSocket&) = delete;

    // False means the ring is full; the caller re-offers rather than this blocking.
    bool submit(const BridgeSendTask& task);

    using Retire = std::function<void(uint32_t arena, uint32_t pages)>;
    using Deliver = std::function<bool(const BridgeDeliverTask&)>;

    // One pass: retire what the peer took, deliver what arrived. poke_progress() first, since
    // flush completes only our puts and sync is just a barrier (§7.2d).
    uint32_t poll(const Retire& retire, const Deliver& deliver);

    // The consumer is done with those pages, so the peer may reuse the slots.
    void consumed(uint32_t arena, uint32_t pages);

    bool publish_credits();
    uint64_t credit_total(uint32_t arena) const;

    // ROUND TRIP, not one-way: put -> the peer's credit for that frame. Two hosts share no
    // clock epoch, so a one-way figure here would be meaningless.
    const std::vector<uint64_t>& put_to_credit_ns() const;

    struct PassStats {
        uint64_t passes = 0;          // poll() calls that reached the flush
        uint64_t starved = 0;         // of those, the ones that posted nothing
        uint64_t starved_credit = 0;  // a destination ring was full
        uint64_t starved_window = 0;  // in_flight hit the window: flush sooner, do not batch
        uint64_t starved_empty = 0;   // nothing was queued to send
        uint64_t posts = 0;           // FRAMES posted, not puts: coalescing adds `run` at once
        uint64_t payload_puts = 0;    // the operations those frames cost, one per run
        uint64_t trailer_puts = 0;    // the second phase, also one per run
        uint64_t credit_puts = 0;     // coalesced credit puts, NOT frames credited
        uint64_t flushes = 0;
        uint64_t in_flight_sum = 0;
        uint64_t bad_order = 0;  // repeats and strays; AHEAD is normal, not disorder
    };
    const PassStats& pass_stats() const;

    struct Series {
        std::vector<uint32_t> in_flight;  // frames outstanding, one per poll pass
    };
    const Series& series() const;
    void reset_stats();

    std::string barrier();

    bool failed() const;
    std::string first_error() const;
    std::string describe() const;

private:
    EriscH2HSocket();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tt::tt_metal::experimental
