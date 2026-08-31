// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Cross-host clock synchronisation, run before the measurement starts.
//
// Cristian's algorithm over the bootstrap socket, taking the minimum-RTT sample:
//
//     t0 = our clock;  send        -->
//                                  <--  peer replies with t1 = its clock
//     t2 = our clock
//
//     rtt    = t2 - t0
//     offset = t1 - (t0 + rtt/2)
//
// The estimate is only as good as the assumption that the path is symmetric, and its
// error is bounded by half the RTT. So the MINIMUM-RTT sample is kept rather than the
// mean: the sample with the least queueing is the one where the symmetry assumption is
// least abused. The bound is reported alongside the offset, because a one-way hop of
// 3 us measured with a +/- 12 us bound is not a measurement and the table should say so
// rather than print a confident number.
//
// This deliberately does NOT try to be PTP. Hardware timestamping would do far better,
// but it needs NIC support this path cannot assume across both tcp and verbs.
// What is here is honest about its own error, which is the property that matters for
// deciding whether a cross-host hop number can be believed.
#pragma once

#include <cstdint>
#include <string>

namespace tt::tt_metal::experimental {

struct ClockSync {
    int64_t offset_ns = 0;        // add to a PEER timestamp to express it on our clock
    uint64_t min_rtt_ns = 0;
    uint64_t uncertainty_ns = 0;  // half the minimum RTT: the bound on offset_ns
    uint32_t samples = 0;
    bool same_host = false;
    bool valid = false;
    std::string error;

    // Converts a peer timestamp to our timeline. Refuses rather than guesses when the
    // sync failed: a silently unconverted peer timestamp would produce a hop duration in
    // the hundreds of seconds, which looks like a hang rather than like a bug.
    bool to_local(uint64_t peer_ts, uint64_t& out) const {
        if (!valid) {
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

// Both sides must call this at the same point in the sequence, and they must pass
// opposite values of `initiator` -- one probes, the other answers. Deriving it from
// is_server rather than from a separate flag is what keeps the two from both probing and
// deadlocking.
ClockSync sync_clocks(int oob_fd, bool initiator, bool same_host, uint32_t samples = 64);

}  // namespace tt::tt_metal::experimental
