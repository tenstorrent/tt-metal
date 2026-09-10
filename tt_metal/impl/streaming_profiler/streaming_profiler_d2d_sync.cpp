// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/streaming_profiler_d2d_sync.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <tt-logger/tt-logger.hpp>

#include "impl/streaming_profiler/spsc_packet.h"
#include "tools/profiler/sync/eth_wallclock_sync_refclk.hpp"

namespace tt::tt_metal::streaming_profiler {

void D2dSyncConsumer::on_clock(const ClockSample& s) {
    if (s.kind == PP_CLOCK_LOCAL_REFCLK) {
        LocalState& l = local_[s.dev];
        l.fit.add(l.unwrap.full(s.value24), s.ts);
    } else if (s.kind == PP_CLOCK_LINK_REFCLK) {
        LinkState& k = link_[{s.dev, s.lane / profiler::kSpscNRiscDecode}];
        k.samples.push_back(LinkSample{.wall = s.ts, .refclk = k.unwrap.full(s.value24)});
    } else {
        dropped_kind_++;
    }
}

void D2dSyncConsumer::on_capture_end(const CaptureContext& ctx) {
    // ---- local half: per device, the AICLK that actually applied, from the 1 ms bucket slopes -----------------
    for (auto& [dev, l] : local_) {
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
        if (nb == 0) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] d2d sync device {}: {} local clock samples but no fittable 1 ms bucket",
                dev,
                l.fit.n_total);
            continue;
        }
        const double mean = ssum / static_cast<double>(nb);
        const double to_ghz = LocalClockFit::kRefclkHz * 1e-9;  // wall ticks per refclk tick -> GHz
        log_info(
            tt::LogMetal,
            "[streaming profiler] d2d sync device {}: local clock {} samples in {} x 1 ms buckets; applied AICLK mean "
            "{:.5f} GHz (bucket min {:.5f}, max {:.5f}), spread {:.1f} ppm",
            dev,
            l.fit.n_total,
            nb,
            mean * to_ghz,
            smin * to_ghz,
            smax * to_ghz,
            mean > 0.0 ? (smax - smin) / mean * 1e6 : 0.0);
    }

    // ---- link half: each boot-time eth sync, sender (round start/end) and receiver (arrival) paired by round ---
    const auto core_index = [&](uint32_t dev, const CoreCoord& eth) -> int64_t {
        if (dev >= ctx.devices.size()) {
            return -1;
        }
        const auto& lanes = ctx.devices[dev].lanes;
        for (size_t ci = 0; (ci + 1) * profiler::kSpscNRiscDecode <= lanes.size(); ci++) {
            if (lanes[ci * profiler::kSpscNRiscDecode].logical == eth) {
                return static_cast<int64_t>(ci);
            }
        }
        return -1;
    };
    for (const CaptureContext::Link& L : ctx.links) {
        const int64_t ca = core_index(L.dev_a, L.eth_a);
        const int64_t cb = core_index(L.dev_b, L.eth_b);
        if (ca < 0 || cb < 0) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] d2d sync link chip {} -> chip {}: an eth end is not in the decode roster",
                L.chip_a,
                L.chip_b);
            continue;
        }
        const auto ia = link_.find({L.dev_a, static_cast<uint32_t>(ca)});
        const auto ib = link_.find({L.dev_b, static_cast<uint32_t>(cb)});
        const size_t n_snd = ia == link_.end() ? 0 : ia->second.samples.size();
        const size_t n_rcv = ib == link_.end() ? 0 : ib->second.samples.size();
        if (n_snd < 2 || n_rcv < 1) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] d2d sync link chip {} eth({},{}) -> chip {} eth({},{}): no link stamps drained "
                "({} sender, {} receiver)",
                L.chip_a,
                L.eth_a.x,
                L.eth_a.y,
                L.chip_b,
                L.eth_b.x,
                L.eth_b.y,
                n_snd,
                n_rcv);
            continue;
        }
        const auto& snd_s = ia->second.samples;
        const auto& rcv_s = ib->second.samples;
        // Sender: two stamps per round (start, end); receiver: one (arrival). t0/t2 and t1 with a refclk each.
        const size_t n = std::min(snd_s.size() / 2, rcv_s.size());
        std::vector<eth_sync::EthSyncSample> snd(n), rcv(n);
        for (size_t i = 0; i < n; i++) {
            const LinkSample& s0 = snd_s[2 * i];
            const LinkSample& s2 = snd_s[2 * i + 1];
            const LinkSample& r1 = rcv_s[i];
            eth_sync::EthSyncSample& a = snd[i];
            a.t0_hi = static_cast<uint32_t>(s0.wall >> 32);
            a.t0_lo = static_cast<uint32_t>(s0.wall);
            a.t2_hi = static_cast<uint32_t>(s2.wall >> 32);
            a.t2_lo = static_cast<uint32_t>(s2.wall);
            a.rc_hi = static_cast<uint32_t>(s0.refclk >> 32);
            a.rc_lo = static_cast<uint32_t>(s0.refclk);
            eth_sync::EthSyncSample& b = rcv[i];
            b.t1_hi = static_cast<uint32_t>(r1.wall >> 32);
            b.t1_lo = static_cast<uint32_t>(r1.wall);
            b.rc_hi = static_cast<uint32_t>(r1.refclk >> 32);
            b.rc_lo = static_cast<uint32_t>(r1.refclk);
        }
        const eth_sync::RefclkSolution sol = eth_sync::solve_refclk_domain(snd, rcv);
        if (!sol.valid) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] d2d sync link chip {} eth({},{}) -> chip {} eth({},{}): {} rounds, refclk solve "
                "invalid (sender ratio valid {}, receiver ratio valid {})",
                L.chip_a,
                L.eth_a.x,
                L.eth_a.y,
                L.chip_b,
                L.eth_b.x,
                L.eth_b.y,
                n,
                sol.k_snd.valid,
                sol.k_rcv.valid);
            continue;
        }
        log_info(
            tt::LogMetal,
            "[streaming profiler] d2d sync link chip {} eth({},{}) -> chip {} eth({},{}): {} rounds ({} kept); refclk "
            "domain offset {:.1f} ns, rate {:.3f} ppm, residual rms {:.1f} ns; AICLK/refclk sender {:.4f} ({:+.1f} "
            "ppm vs nominal) receiver {:.4f} ({:+.1f} ppm)",
            L.chip_a,
            L.eth_a.x,
            L.eth_a.y,
            L.chip_b,
            L.eth_b.x,
            L.eth_b.y,
            sol.n_total,
            sol.n_kept,
            sol.offset_ns,
            sol.rate_ppm,
            sol.residual_rms_ns,
            sol.k_snd.k,
            sol.k_snd.ppm_vs_nominal,
            sol.k_rcv.k,
            sol.k_rcv.ppm_vs_nominal);
    }
    if (dropped_kind_ != 0) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] d2d sync: {} PP_CLOCK samples of an unknown kind ignored",
            dropped_kind_);
    }
    // A capture's samples are its own.
    local_.clear();
    link_.clear();
    dropped_kind_ = 0;
}

}  // namespace tt::tt_metal::streaming_profiler
