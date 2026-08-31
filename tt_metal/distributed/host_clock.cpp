// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "host_clock.hpp"

#include <sys/socket.h>

#include <cerrno>
#include <cstring>
#include <sstream>

#include "host_stats.hpp"

namespace tt::tt_metal::experimental {

namespace {

struct Probe {
    uint64_t seq;
    uint64_t peer_now;  // filled by the responder
};

bool xfer(int fd, void* buf, size_t len, bool sending) {
    auto* p = static_cast<uint8_t*>(buf);
    size_t done = 0;
    while (done < len) {
        const ssize_t n = sending ? ::send(fd, p + done, len - done, 0) : ::recv(fd, p + done, len - done, 0);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) {
                continue;
            }
            return false;
        }
        done += static_cast<size_t>(n);
    }
    return true;
}

}  // namespace

std::string ClockSync::describe() const {
    std::ostringstream o;
    if (same_host) {
        return "same host: peer shares this clock, offset is exactly 0";
    }
    if (!valid) {
        return "clock sync FAILED: " + error + " -- cross-host hop timings are not trustworthy";
    }
    o << "offset " << offset_ns << " ns (+/- " << uncertainty_ns << " ns), min RTT " << min_rtt_ns << " ns over "
      << samples << " samples";
    return o.str();
}

ClockSync sync_clocks(int oob_fd, bool initiator, bool same_host, uint32_t samples) {
    ClockSync s;
    s.same_host = same_host;

    if (same_host) {
        // Not an approximation -- both processes read the same hardware clock, so the
        // offset is zero by construction. Estimating it would substitute measurement
        // noise for a known-exact value.
        s.valid = true;
        s.offset_ns = 0;
        s.uncertainty_ns = 0;
        return s;
    }

    if (oob_fd < 0) {
        s.error = "no bootstrap socket";
        return s;
    }

    if (samples == 0) {
        samples = 1;
    }

    if (initiator) {
        uint64_t best_rtt = UINT64_MAX;
        int64_t best_offset = 0;
        uint32_t got = 0;
        for (uint32_t i = 0; i < samples; ++i) {
            Probe p{i, 0};
            const uint64_t t0 = now_ns();
            if (!xfer(oob_fd, &p, sizeof(p), true)) {
                s.error = "sending probe failed";
                return s;
            }
            if (!xfer(oob_fd, &p, sizeof(p), false)) {
                s.error = "receiving probe reply failed";
                return s;
            }
            const uint64_t t2 = now_ns();
            if (t2 < t0) {
                continue;
            }
            const uint64_t rtt = t2 - t0;
            // Keep the MINIMUM-RTT sample, not a running average. The least-queued
            // exchange is the one where "the path is symmetric" is closest to true, and
            // averaging pulls the estimate toward the samples where it is least true.
            if (rtt < best_rtt) {
                best_rtt = rtt;
                best_offset = static_cast<int64_t>(p.peer_now) - static_cast<int64_t>(t0 + rtt / 2);
            }
            ++got;
        }
        if (got == 0 || best_rtt == UINT64_MAX) {
            s.error = "no usable samples";
            return s;
        }
        s.offset_ns = best_offset;
        s.min_rtt_ns = best_rtt;
        s.uncertainty_ns = best_rtt / 2;
        s.samples = got;
        s.valid = true;

        // Tell the responder the result so BOTH sides can convert peer timestamps, and
        // so both print the same numbers. A run where the two hosts disagree about the
        // offset would produce two reports that cannot be reconciled.
        if (!xfer(oob_fd, &s.offset_ns, sizeof(s.offset_ns), true) ||
            !xfer(oob_fd, &s.uncertainty_ns, sizeof(s.uncertainty_ns), true) ||
            !xfer(oob_fd, &s.min_rtt_ns, sizeof(s.min_rtt_ns), true)) {
            s.error = "publishing offset to peer failed";
            s.valid = false;
        }
        return s;
    }

    // Responder: stamp and echo, as fast as possible. Nothing else happens between the
    // read and the timestamp, because any work in between is added directly to the
    // asymmetry the estimate assumes away.
    for (uint32_t i = 0; i < samples; ++i) {
        Probe p{};
        if (!xfer(oob_fd, &p, sizeof(p), false)) {
            s.error = "receiving probe failed";
            return s;
        }
        p.peer_now = now_ns();
        if (!xfer(oob_fd, &p, sizeof(p), true)) {
            s.error = "echoing probe failed";
            return s;
        }
    }

    int64_t offset = 0;
    uint64_t unc = 0, rtt = 0;
    if (!xfer(oob_fd, &offset, sizeof(offset), false) || !xfer(oob_fd, &unc, sizeof(unc), false) ||
        !xfer(oob_fd, &rtt, sizeof(rtt), false)) {
        s.error = "receiving offset from peer failed";
        return s;
    }
    // NEGATED. The initiator computed "add this to a responder timestamp to get initiator
    // time"; from the responder's side the conversion runs the other way. Getting this
    // backwards produces an offset of exactly the wrong sign, which shows up as one host
    // reporting every cross-host hop as negative -- so it is worth stating rather than
    // leaving to be re-derived.
    s.offset_ns = -offset;
    s.uncertainty_ns = unc;
    s.min_rtt_ns = rtt;
    s.samples = samples;
    s.valid = true;
    return s;
}

}  // namespace tt::tt_metal::experimental
