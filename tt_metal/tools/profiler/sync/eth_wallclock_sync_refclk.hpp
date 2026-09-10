// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Solve the link in the REFCLK domain instead of the wall-clock domain.
//
// Why this exists. The wall clock is AICLK, which the 1 kHz DVFS loop moves and which the hardware droop
// response moves far faster still -- at tens of nanoseconds, entirely inside a 1 kHz sync interval. So a
// slope fitted from wall-clock stamps mixes the thing we want (the two chips' crystal ratio, a stable
// quantity) with the thing we do not (AICLK wandering). That is exactly why a slope with a 0.01 ppm
// standard error over a 51 ms window disagrees with a measurement 30 s later by several ppm: the fit is
// accurate about the window it saw, and stale.
//
// Each end's 50 MHz counter does not move under DVFS or droop. Fitting wall against refclk across the
// burst recovers the AICLK rate that ACTUALLY applied, and converting the stamps through it puts both
// ends into their own real-time domains before they are compared. What remains between the two ends is
// then the crystal ratio alone.
//
// The wall clock stays the WIRE observable throughout. Refclk quantises at 20 ns against the wall clock's
// 0.74 ns, so stamping refclk over the link would throw away more than an order of magnitude of closure.
// It is used only for the LOCAL conversion, where its own quantisation is divided down by the burst.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "tools/profiler/sync/eth_wallclock_sync_solve.hpp"
#include "tools/profiler/sync/eth_wallclock_sync_types.hpp"

namespace tt::tt_metal::eth_sync {

// One end's AICLK rate, measured against its own refclk across the burst.
struct LocalRatio {
    bool valid = false;
    double k = 0.0;  // AICLK cycles per refclk tick; nominal is 27.0 at 1.35 GHz / 50 MHz
    uint64_t w0 = 0, rc0 = 0;
    double resid_rms = 0.0;  // cycles about the fitted line: how steady AICLK was during the burst
    double ppm_vs_nominal = 0.0;
    size_t n = 0;
    size_t n_outliers = 0;  // points 3-sigma trimming removed
};

inline LocalRatio fit_local_ratio(const std::vector<EthSyncSample>& v, bool sender) {
    LocalRatio r;
    std::vector<std::pair<uint64_t, uint64_t>> pts;  // (wall, refclk) for THIS end
    pts.reserve(v.size());
    for (const auto& s : v) {
        const uint64_t w = sender ? ((static_cast<uint64_t>(s.t2_hi) << 32) | s.t2_lo)
                                  : ((static_cast<uint64_t>(s.t1_hi) << 32) | s.t1_lo);
        const uint64_t rc = (static_cast<uint64_t>(s.rc_hi) << 32) | s.rc_lo;
        if (w == 0 || rc == 0) {
            continue;  // a partial run leaves zeros past the samples actually taken
        }
        pts.emplace_back(w, rc);
    }
    if (pts.size() < 8) {
        return r;
    }
    // Centre on the first point: at absolute magnitudes (~1e13 cycles) the products lose the precision
    // the slope needs -- the same trap the wall-domain solve documents.
    r.w0 = pts.front().first;
    r.rc0 = pts.front().second;
    const double n = static_cast<double>(pts.size());
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (const auto& p : pts) {
        const double x = static_cast<double>(p.second - r.rc0);
        const double y = static_cast<double>(p.first - r.w0);
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    const double den = n * sxx - sx * sx;
    if (std::abs(den) < 1e-9) {
        return r;
    }
    r.k = (n * sxy - sx * sy) / den;
    double inter = (sy - r.k * sx) / n;

    // OUTLIER REJECTION. Without it a single bad point owns the slope: an unguarded fit produced a
    // -3.79 ppm excursion on one repeat that the wall-domain rate -- reading the SAME round trips --
    // did not see at all, which is how we know it was the fit and not the clock. solve() filters for
    // exactly this reason; this fit must too. Two passes of 3-sigma trimming, refitting each time.
    std::vector<char> keep(pts.size(), 1);
    for (int pass = 0; pass < 2; pass++) {
        double ss0 = 0;
        size_t nk = 0;
        for (size_t i = 0; i < pts.size(); i++) {
            if (!keep[i]) {
                continue;
            }
            const double x = static_cast<double>(pts[i].second - r.rc0);
            const double res = static_cast<double>(pts[i].first - r.w0) - (inter + r.k * x);
            ss0 += res * res;
            nk++;
        }
        if (nk < 8) {
            break;
        }
        const double sig = std::sqrt(ss0 / static_cast<double>(nk));
        if (sig <= 0.0) {
            break;
        }
        size_t dropped = 0;
        for (size_t i = 0; i < pts.size(); i++) {
            if (!keep[i]) {
                continue;
            }
            const double x = static_cast<double>(pts[i].second - r.rc0);
            const double res = static_cast<double>(pts[i].first - r.w0) - (inter + r.k * x);
            if (std::abs(res) > 3.0 * sig) {
                keep[i] = 0;
                dropped++;
            }
        }
        if (dropped == 0) {
            break;
        }
        double ax = 0, ay = 0, axx = 0, axy = 0, an = 0;
        for (size_t i = 0; i < pts.size(); i++) {
            if (!keep[i]) {
                continue;
            }
            const double x = static_cast<double>(pts[i].second - r.rc0);
            const double y = static_cast<double>(pts[i].first - r.w0);
            ax += x;
            ay += y;
            axx += x * x;
            axy += x * y;
            an += 1.0;
        }
        const double d2 = an * axx - ax * ax;
        if (std::abs(d2) < 1e-9) {
            break;
        }
        r.k = (an * axy - ax * ay) / d2;
        inter = (ay - r.k * ax) / an;
    }

    double ss = 0;
    size_t nkept = 0;
    for (size_t i = 0; i < pts.size(); i++) {
        if (!keep[i]) {
            continue;
        }
        const double x = static_cast<double>(pts[i].second - r.rc0);
        const double res = static_cast<double>(pts[i].first - r.w0) - (inter + r.k * x);
        ss += res * res;
        nkept++;
    }
    r.n_outliers = pts.size() - nkept;
    r.resid_rms = nkept ? std::sqrt(ss / static_cast<double>(nkept)) : 0.0;
    r.ppm_vs_nominal = (r.k / 27.0 - 1.0) * 1e6;
    r.n = pts.size();
    r.valid = true;
    return r;
}

struct RefclkSolution {
    bool valid = false;
    size_t n_total = 0, n_kept = 0;
    double offset_ticks = 0.0;  // receiver minus sender at mid_ref, in 20 ns refclk ticks
    double offset_ns = 0.0;
    double rate_ppm = 0.0;  // drift of that offset per unit time: the crystal ratio, DVFS removed
    double residual_rms_ns = 0.0;
    LocalRatio k_snd, k_rcv;
};

// Same shape as solve(): keep the fastest trips (a slow trip is a biased estimate of the instant, not a
// wrong measurement), then regress offset against time for the rate.
inline RefclkSolution solve_refclk_domain(
    const std::vector<EthSyncSample>& snd, const std::vector<EthSyncSample>& rcv, double keep_frac = 0.25) {
    RefclkSolution s;
    s.k_snd = fit_local_ratio(snd, /*sender=*/true);
    s.k_rcv = fit_local_ratio(rcv, /*sender=*/false);
    if (!s.k_snd.valid || !s.k_rcv.valid || s.k_snd.k <= 0 || s.k_rcv.k <= 0) {
        return s;
    }
    const size_t n = std::min(snd.size(), rcv.size());

    struct RT {
        double off, mid;
        uint64_t rtt;
    };
    std::vector<RT> rts;
    rts.reserve(n);
    for (size_t i = 0; i < n; i++) {
        const uint64_t t0 = (static_cast<uint64_t>(snd[i].t0_hi) << 32) | snd[i].t0_lo;
        const uint64_t t2 = (static_cast<uint64_t>(snd[i].t2_hi) << 32) | snd[i].t2_lo;
        const uint64_t t1 = (static_cast<uint64_t>(rcv[i].t1_hi) << 32) | rcv[i].t1_lo;
        if (t0 == 0 || t1 == 0 || t2 == 0 || t2 < t0) {
            continue;
        }
        // Each end's wall stamp, expressed in its OWN refclk ticks via its OWN measured rate.
        const double r0 = static_cast<double>(t0 - s.k_snd.w0) / s.k_snd.k;
        const double r2 = static_cast<double>(t2 - s.k_snd.w0) / s.k_snd.k;
        const double r1 = static_cast<double>(t1 - s.k_rcv.w0) / s.k_rcv.k;
        const double mid = 0.5 * (r0 + r2);
        // The constant (rc0_rcv - rc0_snd) is the chips' refclk phase difference; it rides in the offset,
        // which is what we are measuring, and cancels out of the RATE entirely.
        const double off = (static_cast<double>(s.k_rcv.rc0) + r1) - (static_cast<double>(s.k_snd.rc0) + mid);
        rts.push_back(RT{off, mid, t2 - t0});
    }
    s.n_total = rts.size();
    if (rts.size() < 4) {
        return s;
    }
    std::sort(rts.begin(), rts.end(), [](const RT& a, const RT& b) { return a.rtt < b.rtt; });
    size_t keep = static_cast<size_t>(static_cast<double>(rts.size()) * keep_frac);
    keep = std::max<size_t>(keep, 4);
    keep = std::min(keep, rts.size());
    rts.resize(keep);
    s.n_kept = keep;

    double mid0 = rts[0].mid;
    for (const auto& r : rts) {
        mid0 = std::min(mid0, r.mid);
    }
    const double nn = static_cast<double>(rts.size());
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (const auto& r : rts) {
        const double x = r.mid - mid0;
        sx += x;
        sy += r.off;
        sxx += x * x;
        sxy += x * r.off;
    }
    const double den = nn * sxx - sx * sx;
    const double slope = std::abs(den) > 1e-9 ? (nn * sxy - sx * sy) / den : 0.0;
    const double inter = (sy - slope * sx) / nn;
    double ss = 0;
    for (const auto& r : rts) {
        const double res = r.off - (inter + slope * (r.mid - mid0));
        ss += res * res;
    }
    s.residual_rms_ns = std::sqrt(ss / nn) * 20.0;  // ticks -> ns
    s.offset_ticks = inter;
    s.offset_ns = inter * 20.0;
    s.rate_ppm = slope * 1e6;  // ticks of offset per tick of elapsed time
    s.valid = true;
    return s;
}

}  // namespace tt::tt_metal::eth_sync
