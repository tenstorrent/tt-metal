// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "impl/streaming_profiler/streaming_profiler_consumer.hpp"
#include "impl/streaming_profiler/streaming_profiler_decode.hpp"

namespace tt::tt_metal::streaming_profiler {

// The local half of the device<->device sync: one chip's AICLK wall clock against its eth tile's free-running
// 50 MHz refclk, from the idle-eth tracker's PP_CLOCK(LOCAL) samples (one every 3 us).
//
// Fitted in 1 ms buckets, each RELATIVE TO ITS OWN FIRST SAMPLE. A bucket's slope is the number of wall ticks per
// refclk tick that ACTUALLY applied over that millisecond -- the AICLK rate under DVFS, not the nominal one -- and a
// 1 ms bucket holds ~330 samples, about one DVFS excursion (a straight line through a 0.5 s bucket left tens of
// microseconds of residual that was the model's error, not the sync's). Anchoring each bucket at its own origin
// removes the lever arm a session-origin intercept would have (a 1 ppm slope error 30 s in is 30 us at the origin).
struct LocalClockFit {
    static constexpr double kRefclkHz = 50e6;
    static constexpr uint64_t kBucketTicks = static_cast<uint64_t>(kRefclkHz * 0.001);  // 1 ms of refclk

    struct Accum {
        bool anchored = false;
        double ax = 0.0, ay = 0.0;  // this bucket's own origin: (refclk, wall)
        long double sx = 0, sy = 0, sxx = 0, sxy = 0;
        uint64_t n = 0;
        // Wall ticks per refclk tick over this bucket: the applied AICLK / 50 MHz.
        double slope() const {
            if (n < 2) {
                return 0.0;
            }
            const long double nn = static_cast<long double>(n);
            const long double den = nn * sxx - sx * sx;
            return den > 0 ? static_cast<double>((nn * sxy - sx * sy) / den) : 0.0;
        }
        double intercept() const {
            if (n < 2) {
                return 0.0;
            }
            return static_cast<double>((sy - static_cast<long double>(slope()) * sx) / static_cast<long double>(n));
        }
        // The fitted line, both ways, through this bucket's own anchor so a conversion never leaves the interval.
        double wall_of_refclk(double r) const { return ay + intercept() + slope() * (r - ax); }
        double refclk_of_wall(double w) const { return ax + (w - ay - intercept()) / slope(); }
        void add(double x, double y) {
            if (!anchored) {
                ax = x;
                ay = y;
                anchored = true;
            }
            const long double dx = x - ax, dy = y - ay;
            sx += dx;
            sy += dy;
            sxx += dx * dx;
            sxy += dx * dy;
            n++;
        }
    };

    // Buckets keyed by refclk_ticks / kBucketTicks; a std::map because the tracker can be paused by a ship.
    std::map<uint64_t, Accum> buckets;
    uint64_t n_total = 0;

    void add(uint64_t refclk_ticks, uint64_t wall_ticks) {
        buckets[refclk_ticks / kBucketTicks].add(static_cast<double>(refclk_ticks), static_cast<double>(wall_ticks));
        n_total++;
    }
};

// Reassembles a 24-bit refclk value stream (the PP_CLOCK payload) into 64-bit ticks. A source samples at least every
// few ms, far inside the 0.335 s a 24-bit wrap takes at 50 MHz, so a drop of more than half the range is a wrap.
struct RefclkUnwrap {
    bool seeded = false;
    uint32_t last = 0;
    uint64_t wraps = 0;
    uint64_t full(uint32_t v24) {
        if (seeded && v24 < last && (last - v24) > (1u << 23)) {
            wraps++;
        }
        seeded = true;
        last = v24;
        return (wraps << 24) | v24;
    }
};

// Device<->device sync from the PP_CLOCK samples the idle-eth pushers carry, and the correction it publishes.
//
// LOCAL samples feed one LocalClockFit per device. LINK samples (the boot-time eth sync rounds: sender round start
// and end, receiver arrival) are paired by round and solved refclk against refclk, so DVFS on either wall clock
// cannot enter the link solve. From those the consumer publishes, per chip, a time-indexed correction to the baked
// host anchor every Record carries (SyncCorrections; Record::host_time composes it):
//
//   root chip r:      host(T) = H_r + (R_r(T) - R_r(A_r)) * P_r         (static host anchor o applied-AICLK term)
//   non-root chip c:  host(T) = H_r + (link(R_c(T)) - R_r(A_r)) * P_r   (the same, on the root's timeline)
//
// where R_x(T) inverts the 1 ms bucket holding wall tick T, A_x/H_x are the chip's boot anchor (tick, host ns), P_x
// its refclk period taken as k_mean/hz so it is consistent with that anchor, and link() maps c's refclk onto r's by
// the solved offset and rate about the burst midpoint. Published incrementally for live sinks, finally at capture end.
// Runs entirely on its consumer's thread.
class D2dSyncConsumer {
public:
    void on_attach(const CaptureContext& ctx);
    void on_clock(const ClockSample& s);
    void on_capture_end(const CaptureContext& ctx);

private:
    struct LinkSample {
        uint64_t wall = 0;
        uint64_t refclk = 0;
    };
    struct LocalState {
        LocalClockFit fit;
        RefclkUnwrap unwrap;
    };
    struct LinkState {
        RefclkUnwrap unwrap;
        std::vector<LinkSample> samples;  // in emission order
    };
    // A solved link: receiver refclk = sender refclk + offset + rate * (sender refclk - mid).
    struct LinkSolution {
        bool ok = false;
        uint32_t dev_snd = 0, dev_rcv = 0;
        double offset_ticks = 0.0, rate = 0.0, mid = 0.0;
        double offset_ns = 0.0, rate_ppm = 0.0, residual_rms_ns = 0.0;
        size_t rounds = 0, kept = 0;
    };
    // A chip's refclk frame: its anchor tick's refclk and its refclk period, once its buckets allow.
    struct Frame {
        bool ok = false;
        double refclk_at_anchor = 0.0;
        double period_ns = 0.0;
        double k_mean = 0.0;
    };

    int64_t core_index(uint32_t dev, const CoreCoord& eth) const;
    void try_solve_links(bool final);
    Frame frame_of(uint32_t dev) const;
    void publish_all(bool final);
    void log_summary() const;

    CaptureContext ctx_;
    std::map<uint32_t, LocalState> local_;                     // device index -> local fit
    std::map<std::pair<uint32_t, uint32_t>, LinkState> link_;  // (device index, core index) -> link stamps
    std::vector<LinkSolution> solved_;                         // per ctx_.links index
    uint64_t dropped_kind_ = 0;
    size_t buckets_at_publish_ = 0;
    static constexpr size_t kPublishEveryBuckets = 100;  // ~100 ms of tracker time between live publications
    static constexpr size_t kLinkRounds = 240;           // the boot-time burst
};

}  // namespace tt::tt_metal::streaming_profiler
