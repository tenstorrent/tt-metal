// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/streaming_profiler_d2d_sync.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <tt-logger/tt-logger.hpp>
#include <cstdlib>
#include <cstdio>

#include "impl/streaming_profiler/spsc_packet.h"
#include "impl/streaming_profiler/streaming_profiler_sync_correction.hpp"
#include "tools/profiler/sync/eth_wallclock_sync_refclk.hpp"

namespace tt::tt_metal::streaming_profiler {

namespace {
// The frequency the records were baked with (record_consts rounds the same way), so base and correction agree.
uint32_t baked_hz(const DeviceClock& k) {
    return static_cast<uint32_t>(
        std::clamp<int64_t>(std::llround(k.frequency_ghz * 1e9), 1, std::numeric_limits<uint32_t>::max()));
}
}  // namespace

void D2dSyncConsumer::on_attach(const CaptureContext& ctx) {
    ctx_ = ctx;
    local_.clear();
    link_.clear();
    solved_.assign(ctx.links.size(), LinkSolution{});
    dropped_kind_ = 0;
    buckets_at_publish_ = 0;
    // A new capture: its corrections start from nothing.
    for (const CaptureContext::Device& d : ctx.devices) {
        SyncCorrections::clear(d.chip_id);
    }
}

void D2dSyncConsumer::on_clock(const ClockSample& s) {
    if (s.kind == PP_CLOCK_LOCAL_REFCLK) {
        LocalState& l = local_[s.dev];
        l.fit.add(l.unwrap.full(s.value24), s.ts);
        // Live publication: every ~100 ms of tracker time across the devices.
        size_t nb = 0;
        for (const auto& [dev, st] : local_) {
            nb += st.fit.buckets.size();
        }
        if (nb >= buckets_at_publish_ + kPublishEveryBuckets) {
            buckets_at_publish_ = nb;
            try_solve_links(/*final=*/false);
            publish_all(/*final=*/false);
        }
    } else if (s.kind == PP_CLOCK_LINK_REFCLK) {
        LinkState& k = link_[{s.dev, s.lane / profiler::kSpscNRiscDecode}];
        k.samples.push_back(LinkSample{.wall = s.ts, .refclk = k.unwrap.full(s.value24)});
    } else {
        dropped_kind_++;
    }
}

int64_t D2dSyncConsumer::core_index(uint32_t dev, const CoreCoord& eth) const {
    if (dev >= ctx_.devices.size()) {
        return -1;
    }
    const auto& lanes = ctx_.devices[dev].lanes;
    for (size_t ci = 0; (ci + 1) * profiler::kSpscNRiscDecode <= lanes.size(); ci++) {
        if (lanes[ci * profiler::kSpscNRiscDecode].logical == eth) {
            return static_cast<int64_t>(ci);
        }
    }
    return -1;
}

// Sender: two stamps per round (start, end); receiver: one (arrival). Each stamp carries the wall tick and a refclk
// reading, which is exactly EthSyncSample -- t0/t2 and t1 with an rc -- so the checkpoint-1 solver runs unchanged.
void D2dSyncConsumer::try_solve_links(bool final) {
    for (size_t li = 0; li < ctx_.links.size(); li++) {
        LinkSolution& out = solved_[li];
        if (out.ok) {
            continue;
        }
        const CaptureContext::Link& L = ctx_.links[li];
        const int64_t ca = core_index(L.dev_a, L.eth_a);
        const int64_t cb = core_index(L.dev_b, L.eth_b);
        if (ca < 0 || cb < 0) {
            continue;
        }
        const auto ia = link_.find({L.dev_a, static_cast<uint32_t>(ca)});
        const auto ib = link_.find({L.dev_b, static_cast<uint32_t>(cb)});
        const size_t n_snd = ia == link_.end() ? 0 : ia->second.samples.size();
        const size_t n_rcv = ib == link_.end() ? 0 : ib->second.samples.size();
        const size_t n = std::min(n_snd / 2, n_rcv);
        // Live: wait for the whole burst. Final: take what arrived, if it is enough for a fit at all.
        if (n < (final ? 8u : kLinkRounds)) {
            continue;
        }
        const auto& snd_s = ia->second.samples;
        const auto& rcv_s = ib->second.samples;
        std::vector<eth_sync::EthSyncSample> snd(n), rcv(n);
        long double mid_acc = 0;
        for (size_t i = 0; i < n; i++) {
            const LinkSample& s0 = snd_s[2 * i];
            const LinkSample& s2 = snd_s[2 * i + 1];
            const LinkSample& r1 = rcv_s[i];
            eth_sync::EthSyncSample& a = snd[i];
            a.t0_hi = static_cast<uint32_t>(s0.wall >> 32);
            a.t0_lo = static_cast<uint32_t>(s0.wall);
            a.t2_hi = static_cast<uint32_t>(s2.wall >> 32);
            a.t2_lo = static_cast<uint32_t>(s2.wall);
            // The solver's fit_local_ratio pairs the sender's refclk with t2 (round END) and the receiver's with t1
            // (arrival). Pairing the sender's with t0 instead biased the link offset by the full round trip -- the
            // synthetic test caught it as exactly 2x the one-way delay.
            a.rc_hi = static_cast<uint32_t>(s2.refclk >> 32);
            a.rc_lo = static_cast<uint32_t>(s2.refclk);
            eth_sync::EthSyncSample& b = rcv[i];
            b.t1_hi = static_cast<uint32_t>(r1.wall >> 32);
            b.t1_lo = static_cast<uint32_t>(r1.wall);
            b.rc_hi = static_cast<uint32_t>(r1.refclk >> 32);
            b.rc_lo = static_cast<uint32_t>(r1.refclk);
            // The burst midpoint the rate term is about: the mean trip midpoint, in the sender's refclk.
            mid_acc += (static_cast<long double>(s0.refclk) + static_cast<long double>(s2.refclk)) / 2;
        }
        const eth_sync::RefclkSolution sol = eth_sync::solve_refclk_domain(snd, rcv);
        if (!sol.valid) {
            if (final) {
                log_warning(
                    tt::LogMetal,
                    "[streaming profiler] d2d sync link chip {} eth({},{}) -> chip {} eth({},{}): {} rounds, refclk "
                    "solve invalid (sender ratio valid {}, receiver ratio valid {})",
                    L.chip_a,
                    L.eth_a.x,
                    L.eth_a.y,
                    L.chip_b,
                    L.eth_b.x,
                    L.eth_b.y,
                    n,
                    sol.k_snd.valid,
                    sol.k_rcv.valid);
            }
            continue;
        }
        out.ok = true;
        out.dev_snd = L.dev_a;
        out.dev_rcv = L.dev_b;
        out.offset_ticks = sol.offset_ticks;
        out.rate = sol.rate_ppm * 1e-6;
        out.mid = static_cast<double>(mid_acc / static_cast<long double>(n));
        out.offset_ns = sol.offset_ns;
        out.rate_ppm = sol.rate_ppm;
        out.residual_rms_ns = sol.residual_rms_ns;
        out.rounds = sol.n_total;
        out.kept = sol.n_kept;
    }
}

// A chip's refclk frame from its buckets: the refclk at its anchor tick, and its refclk period taken as k_mean / hz
// so the correction is consistent with the anchor the records were baked with.
D2dSyncConsumer::Frame D2dSyncConsumer::frame_of(uint32_t dev) const {
    Frame f;
    const auto it = local_.find(dev);
    if (it == local_.end() || dev >= ctx_.devices.size()) {
        return f;
    }
    const LocalClockFit& fit = it->second.fit;
    // Anchor on the ETH clock: the local fit's wall domain is the eth core's wall clock (the sync kernels run on eth
    // cores), so refclk_at_anchor and period_ns must be taken against the eth anchor, not the worker anchor -- the two
    // tiles keep different wall totals (per-card duty cycle), and mixing them was a ~hours domain error.
    const DeviceClock& clk = ctx_.devices[dev].eth_clock;
    if (clk.frequency_ghz <= 0.0) {
        return f;  // no eth anchor: cannot place this chip's refclk on the host timeline
    }
    const double A = static_cast<double>(clk.anchor_ticks);
    double ksum = 0.0;
    size_t nk = 0;
    const LocalClockFit::Accum* holder = nullptr;  // the bucket whose wall span contains the anchor
    const LocalClockFit::Accum* first = nullptr;
    const LocalClockFit::Accum* last = nullptr;
    for (const auto& [key, b] : fit.buckets) {
        if (b.n < 2 || b.slope() <= 0.0) {
            continue;
        }
        ksum += b.slope();
        nk++;
        if (first == nullptr) {
            first = &b;
        }
        last = &b;
        const double r_lo = static_cast<double>(key * LocalClockFit::kBucketTicks);
        const double r_hi = static_cast<double>((key + 1) * LocalClockFit::kBucketTicks);
        if (holder == nullptr && b.wall_of_refclk(r_lo) <= A && A <= b.wall_of_refclk(r_hi)) {
            holder = &b;
        }
    }
    if (nk == 0) {
        return f;
    }
    if (holder == nullptr) {
        // The anchor precedes or follows every fitted bucket: the nearest line, extended.
        holder = A < first->ay ? first : last;
    }
    f.k_mean = ksum / static_cast<double>(nk);
    f.refclk_at_anchor = holder->refclk_of_wall(A);
    // The refclk period consistent with the anchor: the boot fit measured the AICLK that applied AT the anchor, and
    // the anchor bucket's slope is that same rate in wall ticks per refclk tick, so their ratio is the refclk period
    // (20 ns nominal) independent of whatever DVFS does later. The session mean would skew it by any later change.
    f.period_ns = holder->slope() * 1e9 / static_cast<double>(baked_hz(clk));
    f.ok = true;
    return f;
}

void D2dSyncConsumer::publish_all(bool final) {
    if (local_.empty()) {
        return;
    }
    // Root: the lowest device index with a local fit. Its host anchor is the fleet timeline's.
    const uint32_t root = local_.begin()->first;
    const Frame fr = frame_of(root);
    if (!fr.ok) {
        return;
    }
    const DeviceClock& rclk_eth = ctx_.devices[root].eth_clock;
    if (rclk_eth.frequency_ghz <= 0.0) {
        return;  // root has no eth anchor: nothing to place the fleet timeline against
    }

    // Compose each device's refclk onto the root's along the solved-link tree, so a chip with no DIRECT link to the
    // root still lands on the fleet timeline through its neighbours. A link solves receiver = sender*(1+rate) +
    // (offset - rate*mid), an affine in the sender's refclk, and an affine's inverse and composition are affine, so
    // each reachable device carries one { scale, shift } with root_refclk = scale * dev_refclk + shift. A breadth
    // relaxation over the links (few devices, so O(links^2) is nothing) fills them from the root outward; a device
    // no path reaches keeps its own anchor and the local term alone.
    struct RootXf {
        double scale = 1.0, shift = 0.0;
        bool ok = false;
    };
    std::map<uint32_t, RootXf> to_root;
    to_root[root] = RootXf{1.0, 0.0, true};
    for (bool progress = true; progress;) {
        progress = false;
        for (const LinkSolution& s : solved_) {
            if (!s.ok) {
                continue;
            }
            const double m = 1.0 + s.rate;  // receiver = m * sender + o
            const double o = s.offset_ticks - s.rate * s.mid;
            const auto rs = to_root.find(s.dev_rcv);
            const auto ss = to_root.find(s.dev_snd);
            const bool r_ok = rs != to_root.end() && rs->second.ok;
            const bool s_ok = ss != to_root.end() && ss->second.ok;
            if (s_ok && !r_ok) {
                // root = A_s * sender + B_s, and sender = (receiver - o) / m.
                const RootXf& S = ss->second;
                to_root[s.dev_rcv] = RootXf{S.scale / m, S.shift - S.scale * o / m, true};
                progress = true;
            } else if (r_ok && !s_ok) {
                // root = A_r * receiver + B_r, and receiver = m * sender + o.
                const RootXf& R = rs->second;
                to_root[s.dev_snd] = RootXf{R.scale * m, R.scale * o + R.shift, true};
                progress = true;
            }
        }
    }
    for (const auto& [dev, st] : local_) {
        const Frame fd = frame_of(dev);
        if (!fd.ok) {
            continue;
        }
        // The correction is keyed by HOST TIME, not wall ticks: eth and worker tiles keep different wall-clock
        // totals (per-card duty cycle), so map each segment's eth wall-tick bounds to host ns via the eth anchor.
        // A worker zone then looks the correction up by its own base host ns and gets the cross-chip shift.
        const DeviceClock& eclk = ctx_.devices[dev].eth_clock;
        if (eclk.frequency_ghz <= 0.0) {
            continue;  // no eth anchor: cannot place this chip's correction on the host timeline
        }
        const auto to_host = [&](double eth_tick) {
            return static_cast<double>(eclk.anchor_host_ns) +
                   (eth_tick - static_cast<double>(eclk.anchor_ticks)) / eclk.frequency_ghz;
        };
        const double eth_hz = eclk.frequency_ghz;  // eth wall ticks per ns
        // This chip's refclk onto the root's, from the composed transform (identity for the root). The correction is
        // ABSOLUTE: it brings this chip's zones onto the root's timeline, which cancels the chip's static host-anchor
        // error (the demo's point). What it does NOT carry is the boot-random refclk COUNTER offset: root_refclk(T) -
        // fr.refclk_at_anchor is the root's refclk ELAPSED since the root's anchor, so xf.shift and the ~1e13 counter
        // values cancel. Both chips' host DeviceClocks share one host reference (steady_clock, one process), so a
        // common extrapolation error in fr.refclk_at_anchor is common-mode across chips and drops out of any
        // cross-chip comparison; what survives is each chip's own anchor error + the crystal rate drift (~us).
        const auto xf = to_root.find(dev);
        const bool on_root = xf != to_root.end() && xf->second.ok;
        const double xf_scale = on_root ? xf->second.scale : 1.0;
        const double xf_shift = on_root ? xf->second.shift : 0.0;
        // Work in this chip's own eth-anchor frame so magnitudes stay small (steady_clock ns and refclk counters are
        // ~1e13-1e15). dH_eth: the two chips' eth host anchors differ by ~ms (booted at different instants).
        const double dH_eth = static_cast<double>(rclk_eth.anchor_host_ns - eclk.anchor_host_ns);
        const double eth_anchor = static_cast<double>(eclk.anchor_ticks);
        // LINKED: root-timeline host ns of instant (refclk R, eth wall T) minus this chip's own eth-clock host ns,
        // both in the chip's eth-anchor frame. The counter offset has cancelled; the anchor error and rate drift stay.
        const auto link_corr = [&](double R, double T) -> double {
            const double root_refclk = xf_scale * R + xf_shift;
            const double h_link = dH_eth + (root_refclk - fr.refclk_at_anchor) * fr.period_ns;
            const double h_own = (T - eth_anchor) / eth_hz;
            return h_link - h_own;
        };
        // LOCAL: this chip's OWN refclk placing on its OWN host timeline (no cross-chip link) -- corrects only its own
        // AICLK/DVFS drift relative to the DVFS-immune refclk. error = linked - local is then the pure cross-chip term.
        const auto local_corr = [&](double R, double T) -> double {
            const double h_local = (R - fd.refclk_at_anchor) * fd.period_ns;
            const double h_own = (T - eth_anchor) / eth_hz;
            return h_local - h_own;
        };
        std::vector<SyncSegment> segs, local_segs;
        segs.reserve(st.fit.buckets.size());
        local_segs.reserve(st.fit.buckets.size());
        for (const auto& [key, b] : st.fit.buckets) {
            if (b.n < 2 || b.slope() <= 0.0) {
                continue;
            }
            const double r_lo = static_cast<double>(key * LocalClockFit::kBucketTicks);
            const double r_hi = static_cast<double>((key + 1) * LocalClockFit::kBucketTicks);
            const double T_lo = b.wall_of_refclk(r_lo);
            const double T_hi = b.wall_of_refclk(r_hi);
            if (!(T_hi > T_lo)) {
                continue;
            }
            const double d_lo = link_corr(r_lo, T_lo);
            const double d_hi = link_corr(r_hi, T_hi);
            const double H_lo = to_host(T_lo), H_hi = to_host(T_hi);
            segs.push_back(SyncSegment{
                .ns_lo = static_cast<int64_t>(H_lo),
                .ns_hi = static_cast<int64_t>(H_hi),
                .delta_ns_lo = d_lo,
                .slope = (d_hi - d_lo) / (H_hi - H_lo)});
            const double l_lo = local_corr(r_lo, T_lo);
            const double l_hi = local_corr(r_hi, T_hi);
            local_segs.push_back(SyncSegment{
                .ns_lo = static_cast<int64_t>(H_lo),
                .ns_hi = static_cast<int64_t>(H_hi),
                .delta_ns_lo = l_lo,
                .slope = (l_hi - l_lo) / (H_hi - H_lo)});
        }
        if (!segs.empty()) {
            SyncCorrections::publish(ctx_.devices[dev].chip_id, std::move(segs));
        }
        if (!local_segs.empty()) {
            SyncCorrections::publish_local(ctx_.devices[dev].chip_id, std::move(local_segs));
        }
    }
    (void) final;
}

void D2dSyncConsumer::log_summary() const {
    for (const auto& [dev, l] : local_) {
        size_t nb = 0;
        double smin = std::numeric_limits<double>::max(), smax = 0.0, ssum = 0.0;
        for (const auto& [key, b] : l.fit.buckets) {
            if (b.n < 2) {
                continue;
            }
            const double s = b.slope();
            if (s <= 0.0) {
                continue;
            }
            nb++;
            ssum += s;
            smin = std::min(smin, s);
            smax = std::max(smax, s);
        }
        const uint32_t chip = dev < ctx_.devices.size() ? ctx_.devices[dev].chip_id : dev;
        if (nb == 0) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] d2d sync chip {}: {} local clock samples but no fittable 1 ms bucket",
                chip,
                l.fit.n_total);
            continue;
        }
        const double mean = ssum / static_cast<double>(nb);
        const double to_ghz = LocalClockFit::kRefclkHz * 1e-9;
        const double anchor_ghz = dev < ctx_.devices.size() ? ctx_.devices[dev].clock.frequency_ghz : 0.0;
        log_info(
            tt::LogMetal,
            "[streaming profiler] d2d sync chip {}: local clock {} samples in {} x 1 ms buckets; applied AICLK mean "
            "{:.5f} GHz (bucket min {:.5f}, max {:.5f}; boot anchor {:.5f}), spread {:.1f} ppm; {} correction "
            "segments published",
            chip,
            l.fit.n_total,
            nb,
            mean * to_ghz,
            smin * to_ghz,
            smax * to_ghz,
            anchor_ghz,
            mean > 0.0 ? (smax - smin) / mean * 1e6 : 0.0,
            SyncCorrections::published(chip));
    }
    for (size_t li = 0; li < solved_.size() && li < ctx_.links.size(); li++) {
        const LinkSolution& s = solved_[li];
        const CaptureContext::Link& L = ctx_.links[li];
        if (!s.ok) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] d2d sync link chip {} eth({},{}) -> chip {} eth({},{}): not solved (no or too "
                "few link stamps drained)",
                L.chip_a,
                L.eth_a.x,
                L.eth_a.y,
                L.chip_b,
                L.eth_b.x,
                L.eth_b.y);
            continue;
        }
        log_info(
            tt::LogMetal,
            "[streaming profiler] d2d sync link chip {} eth({},{}) -> chip {} eth({},{}): {} rounds ({} kept); refclk "
            "domain offset {:.1f} ns, rate {:.3f} ppm, residual rms {:.1f} ns",
            L.chip_a,
            L.eth_a.x,
            L.eth_a.y,
            L.chip_b,
            L.eth_b.x,
            L.eth_b.y,
            s.rounds,
            s.kept,
            s.offset_ns,
            s.rate_ppm,
            s.residual_rms_ns);
    }
    if (dropped_kind_ != 0) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] d2d sync: {} PP_CLOCK samples of an unknown kind ignored",
            dropped_kind_);
    }
}

// The hedge next to the Tracy plots: a CSV of the local and linked corrections per chip over time and the
// local-vs-linked error, so the accuracy numbers exist even if the Tracy capture is fiddly. One row per 1 ms
// bucket per chip. Gated on TT_METAL_STREAMING_PROFILER_D2D_CSV=<path>.
void D2dSyncConsumer::dump_csv() const {
    const char* path = std::getenv("TT_METAL_STREAMING_PROFILER_D2D_CSV");
    if (path == nullptr || *path == 0) {
        return;
    }
    std::FILE* f = std::fopen(path, "w");
    if (f == nullptr) {
        log_warning(tt::LogMetal, "[streaming profiler] d2d sync: cannot open CSV {}", path);
        return;
    }
    std::fprintf(f, "chip,wall_tick,local_ns,linked_ns,error_ns\n");
    size_t rows = 0;
    double err_sum = 0.0, err_max = 0.0;
    for (const auto& kv : local_) {
        const uint32_t dev = kv.first;
        const uint32_t chip = dev < ctx_.devices.size() ? ctx_.devices[dev].chip_id : dev;
        for (const auto& bk : kv.second.fit.buckets) {
            const LocalClockFit::Accum& b = bk.second;
            if (b.n < 2 || b.slope() <= 0.0) {
                continue;
            }
            const double Tw = b.wall_of_refclk(static_cast<double>(bk.first * LocalClockFit::kBucketTicks));
            const uint64_t T = static_cast<uint64_t>(Tw);
            const DeviceClock& eclk = ctx_.devices[dev].eth_clock;
            const int64_t H = eclk.frequency_ghz > 0.0
                                  ? static_cast<int64_t>(
                                        static_cast<double>(eclk.anchor_host_ns) +
                                        (Tw - static_cast<double>(eclk.anchor_ticks)) / eclk.frequency_ghz)
                                  : 0;
            const long long loc = static_cast<long long>(SyncCorrections::lookup_local_ns(chip, H));
            const long long lnk = static_cast<long long>(SyncCorrections::lookup_ns(chip, H));
            const long long err = lnk - loc;
            std::fprintf(f, "%u,%llu,%lld,%lld,%lld\n", chip, static_cast<unsigned long long>(T), loc, lnk, err);
            rows++;
            const double ae = static_cast<double>(err < 0 ? -err : err);
            err_sum += ae;
            err_max = std::max(err_max, ae);
        }
    }
    std::fclose(f);
    log_info(
        tt::LogMetal,
        "[streaming profiler] d2d sync CSV: {} rows to {}; |local-linked| mean {:.1f} ns, max {:.1f} ns",
        rows,
        path,
        rows ? err_sum / static_cast<double>(rows) : 0.0,
        err_max);
}

void D2dSyncConsumer::on_capture_end(const CaptureContext& ctx) {
    (void)ctx;
    try_solve_links(/*final=*/true);
    publish_all(/*final=*/true);
    log_summary();
    dump_csv();
    // The published corrections stay for the sinks that write at process end; the next attach starts fresh.
    local_.clear();
    link_.clear();
    dropped_kind_ = 0;
}

}  // namespace tt::tt_metal::streaming_profiler
