// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Cross-host clock synchronisation, run before the measurement starts.
//
// Cristian's algorithm over the distributed context:
//
//     t0 = our clock;  send  -->
//                            <--  peer replies with t1 = its clock
//     t2 = our clock;   rtt = t2 - t0;   offset = t1 - (t0 + rtt/2)
//
// Only as good as the assumption that the path is symmetric, with error bounded by half the
// RTT. The MINIMUM-RTT sample is kept rather than the mean -- least queueing is least abuse of
// that assumption -- and uncertainty_ns is that half-RTT bound. It is reported alongside the
// offset because a 3 us hop measured to +/- 12 us is not a measurement and the table has to say
// so rather than print a confident number. A same_host sync skips all of this: the offset is
// zero by construction, so estimating it would substitute measurement noise for a known value.
//
// Deliberately NOT PTP -- hardware timestamping needs NIC support this path cannot assume
// across both tcp and verbs. Being honest about its own error is what decides whether a
// cross-host hop number can be believed.
#pragma once

#include <tt-metalium/distributed_context.hpp>

#include <cstdint>
#include <string>

namespace tt::tt_metal::experimental {

struct ClockSync {
    int64_t offset_ns = 0;  // add to a PEER timestamp to express it on our clock
    uint64_t min_rtt_ns = 0;
    uint64_t uncertainty_ns = 0;  // half the minimum RTT: the bound on offset_ns
    uint32_t samples = 0;
    bool same_host = false;
    bool valid = false;
    // Set when this sync covers more than one peer (sync_clocks_to_hub). offset_ns is then one
    // peer's, so it does not convert another's timestamps; uncertainty_ns and valid still cover
    // the whole set.
    bool multi_peer = false;
    std::string error;

    // Converts a peer timestamp to our timeline. Refuses rather than guesses when the sync
    // failed: an unconverted peer timestamp reads as a hang, not a bug. Refuses a multi-peer
    // sync because "which peer's clock" has no answer there.
    bool to_local(uint64_t peer_ts, uint64_t& out) const {
        if (!valid || multi_peer) {
            return false;
        }
        const int64_t v = static_cast<int64_t>(peer_ts) + offset_ns;
        if (v < 0) {
            return false;
        }
        out = static_cast<uint64_t>(v);
        return true;
    }

    std::string describe() const;
};

// `peer` is the rank to sync against. `initiator` must be true on exactly ONE side of the
// pair -- one probes, the other answers; both initiating deadlocks, neither initiating yields
// no samples. Callers derive it from a rank comparison (self < peer). Both sides must call
// this at the same point in their sequence. The exchange runs over the distributed context
// rather than a socket: MPI has no fd to hand out.
ClockSync sync_clocks(
    const tt::tt_metal::distributed::multihost::ContextPtr& ctx,
    tt::tt_metal::distributed::multihost::Rank peer,
    bool initiator,
    bool same_host,
    uint32_t samples = 64);

// rank 0 IS the hub: it initiates against 1..N-1 in ascending order and everyone else answers
// once, so `initiator` needs no negotiation. Sequential -- overlapping probes would put
// queueing delay into the minimum-RTT estimate. Returns the worst-uncertainty peer's sync
// whole so the triple stays self-consistent, sets multi_peer, and returns a failure as-is.
ClockSync sync_clocks_to_hub(
    const tt::tt_metal::distributed::multihost::ContextPtr& ctx, bool same_host, uint32_t samples = 64);

}  // namespace tt::tt_metal::experimental
