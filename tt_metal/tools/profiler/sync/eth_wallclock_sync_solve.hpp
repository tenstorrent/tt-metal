// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host side of the ethernet wall-clock sync: read both ends' samples out of eth L1 and solve for the
// receiver-vs-sender clock relationship.
//
// The solve is deliberately two-stage, because the two unknowns have very different noise behaviour.
//
//   OFFSET comes from filtered samples, not all of them. Queueing and contention only ever ADD delay, and
//   they add it asymmetrically, so a slow round trip is a biased one -- averaging over everything pulls the
//   answer toward whichever direction happened to be busier. Keeping the fastest trips keeps the ones whose
//   midpoint is most nearly the true instant.
//
//   RATE comes from a regression across the whole (filtered) span, because a frequency difference only
//   shows up as offset DRIFT over time. Fitting it needs a long baseline, which is why the caller should
//   spread the samples rather than take them back to back: the same lesson the host<->device sync learned,
//   where back-to-back sampling left a ~360 us baseline and a 1e-4 frequency error that then grew with
//   time since the anchor.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "tools/profiler/sync/eth_wallclock_sync_types.hpp"

namespace tt::tt_metal::eth_sync {

struct Trip {
    uint64_t t0, t1, t2;  // sender send, receiver arrival, sender echo-observed
    int64_t offset;       // t1 - (t0 + t2) / 2, in cycles
    uint64_t rtt;         // t2 - t0
    uint64_t mid;         // (t0 + t2) / 2, the sender-clock instant the offset belongs to
};

struct EthSyncSolution {
    bool valid = false;
    size_t n_total = 0;     // trips recovered from both ends
    size_t n_kept = 0;      // trips surviving the RTT filter
    int64_t offset = 0;     // receiver MINUS sender, at mid_ref, in cycles
    uint64_t mid_ref = 0;   // sender-clock instant the offset is quoted at
    double rate = 1.0;      // receiver cycles per sender cycle (1.0 == identical rate)
    uint64_t rtt_min = 0;
    uint64_t rtt_med = 0;
    int64_t offset_spread = 0;  // max-min offset across kept trips: the honest error bar
    double residual_rms = 0.0;  // cycles about the fitted line
};

// Pair the two ends by index. Both kernels write exactly one sample per iteration, so index i on the
// sender is the same round trip as index i on the receiver -- the pairing is positional BY CONSTRUCTION
// here (one array each, filled in lockstep), not an assumption about interleaved event streams.
inline std::vector<Trip> build_trips(
    const std::vector<EthSyncSample>& snd, const std::vector<EthSyncSample>& rcv, size_t n) {
    std::vector<Trip> trips;
    trips.reserve(n);
    for (size_t i = 0; i < n; i++) {
        const uint64_t t0 = (static_cast<uint64_t>(snd[i].t0_hi) << 32) | snd[i].t0_lo;
        const uint64_t t2 = (static_cast<uint64_t>(snd[i].t2_hi) << 32) | snd[i].t2_lo;
        const uint64_t t1 = (static_cast<uint64_t>(rcv[i].t1_hi) << 32) | rcv[i].t1_lo;
        if (t0 == 0 || t1 == 0 || t2 == 0 || t2 < t0) {
            continue;  // a partial run leaves zeros past n_samples; a wrapped pair is not usable
        }
        const uint64_t mid = t0 + (t2 - t0) / 2;
        trips.push_back(Trip{t0, t1, t2, static_cast<int64_t>(t1) - static_cast<int64_t>(mid), t2 - t0, mid});
    }
    return trips;
}

// keep_frac: fraction of the fastest trips to keep (0.25 == the quickest quarter). The rest are dropped as
// delayed, not as wrong -- a slow trip is still a true measurement of a slower path, just a biased estimate
// of the instant.
inline EthSyncSolution solve(std::vector<Trip> trips, double keep_frac = 0.25) {
    EthSyncSolution s;
    s.n_total = trips.size();
    if (trips.size() < 4) {
        return s;
    }

    std::sort(trips.begin(), trips.end(), [](const Trip& a, const Trip& b) { return a.rtt < b.rtt; });
    s.rtt_min = trips.front().rtt;
    s.rtt_med = trips[trips.size() / 2].rtt;

    size_t keep = static_cast<size_t>(static_cast<double>(trips.size()) * keep_frac);
    keep = std::max<size_t>(keep, 4);
    keep = std::min(keep, trips.size());
    trips.resize(keep);
    s.n_kept = keep;

    int64_t lo = trips[0].offset, hi = trips[0].offset;
    for (const auto& t : trips) {
        lo = std::min(lo, t.offset);
        hi = std::max(hi, t.offset);
    }
    s.offset_spread = hi - lo;

    // Least squares of offset against the sender-clock instant, centred so the fit is done in small
    // numbers: at absolute wall-clock magnitudes (~1e13) the products overflow double's exact range and
    // the slope comes out as noise. The same centring the host<->device fit needs.
    uint64_t mid0 = trips[0].mid;
    for (const auto& t : trips) {
        mid0 = std::min(mid0, t.mid);
    }
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    const double n = static_cast<double>(trips.size());
    for (const auto& t : trips) {
        const double x = static_cast<double>(t.mid - mid0);
        const double y = static_cast<double>(t.offset);
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    const double denom = n * sxx - sx * sx;
    double slope = 0.0;
    if (std::abs(denom) > 1e-9) {
        slope = (n * sxy - sx * sy) / denom;
    }
    const double intercept = (sy - slope * sx) / n;

    // offset(t) = intercept + slope * (t - mid0). A non-zero slope IS the rate difference: the receiver
    // clock gains `slope` cycles per sender cycle, so its rate is (1 + slope).
    s.rate = 1.0 + slope;
    s.mid_ref = mid0;
    s.offset = static_cast<int64_t>(std::llround(intercept));

    double ss = 0;
    for (const auto& t : trips) {
        const double x = static_cast<double>(t.mid - mid0);
        const double pred = intercept + slope * x;
        const double r = static_cast<double>(t.offset) - pred;
        ss += r * r;
    }
    s.residual_rms = std::sqrt(ss / n);
    s.valid = true;
    return s;
}

}  // namespace tt::tt_metal::eth_sync
