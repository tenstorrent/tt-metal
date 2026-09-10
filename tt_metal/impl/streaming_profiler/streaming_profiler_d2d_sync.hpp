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

// Device<->device sync from the PP_CLOCK samples the idle-eth pushers carry -- the local half above per device, and
// the link half: the boot-time eth sync rounds, whose sender (round start and end) and receiver (arrival) stamps are
// paired by round and solved refclk against refclk, so DVFS on either chip's wall clock cannot enter the link solve.
// Runs on its consumer's own thread: on_clock() for every routed sample, on_capture_end() once per producer.
class D2dSyncConsumer {
public:
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
    std::map<uint32_t, LocalState> local_;                     // device index -> local fit
    std::map<std::pair<uint32_t, uint32_t>, LinkState> link_;  // (device index, core index) -> link stamps
    uint64_t dropped_kind_ = 0;                                // samples of a kind this build does not know
};

}  // namespace tt::tt_metal::streaming_profiler
