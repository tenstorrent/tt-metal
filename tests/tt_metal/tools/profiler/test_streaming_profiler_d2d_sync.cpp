// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host-only check of the d2d sync's placement against a synthetic truth model, no device needed.
//
// A three-chip CHAIN: root(0) -- mid(1) -- leaf(2), with solved links only for (0,1) and (1,2). The leaf has NO
// direct link to the root, so it can only reach the host through the two hops composed -- that composition is what
// this test exercises beyond the single-link case. The chips' refclks run at exactly 50 MHz with known offsets; their
// AICLKs are known functions of true time (chip 0 drops one DVFS step mid-session; chips 1 and 2 steady); the host
// series ties the root's refclk to a TSC of known rate. The placement every record converts through
// (sync.map().lookup_tsc on the record's eth wall tick) must recover TRUE host time in every case:
//   (a) chip 0 before its switch;
//   (b) chip 0 after its switch: the run boundary must be placed where the two lines meet;
//   (c) chip 1 (one hop): only the 0-1 link places it;
//   (d) chip 2 (two hops): only 0-1 composed with 1-2 places it, refclk offset and all.
// The root-refclk placement (the d2d level, which the host series never enters) and the steady_clock view
// (tsc_to_mono_ns through a known segment) are checked alongside.
//
// Chip 1 is a receiver (of 0-1) AND a sender (of 1-2), on two DIFFERENT eth cores -- exactly as real hardware, where
// each link owns its own eth core -- so its two stamp streams stay separate.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "impl/streaming_profiler/spsc_packet.h"
#include "impl/streaming_profiler/streaming_profiler_sync_engine.hpp"
#include "impl/streaming_profiler/streaming_profiler_sync_devices.hpp"
#include "impl/streaming_profiler/streaming_profiler_placement_map.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_metal::streaming_profiler;
namespace api = tt::tt_metal::experimental::streaming_profiler;

static int g_fail = 0;
static void check_near(const char* what, double got, double want, double tol) {
    if (std::fabs(got - want) > tol) {
        std::printf(
            "FAIL %s: got %.3f want %.3f (|err| %.3f > tol %.3f)\n", what, got, want, std::fabs(got - want), tol);
        g_fail++;
    } else {
        std::printf("ok   %s: err %.3f ns (tol %.1f)\n", what, got - want, tol);
    }
}
constexpr double kRefHz = 50e6;
constexpr double kF0 = 1.35e9;           // AICLK at boot on every chip
constexpr double kSlow = 26.875 / 27.0;  // chip 0's AICLK after its DVFS switch: one 1/8 step of the PLL multiple
constexpr double kTauSwitch = 0.300;     // s, when chip 0 slows
constexpr double kOneWay = 1.0e-6;       // s, symmetric link one-way delay
constexpr double kTurn = 350e-9;  // s, the receiver's turnaround: its echo leaves this long after the frame arrived
// chip c refclk = kDref[c] + 50 MHz * tau (the offsets the links must recover); wall origin kW0[c].
constexpr double kDref[3] = {0.0, 1.0e6, 3.0e6};
constexpr double kW0[3] = {1.0e9, 7.0e9, 4.0e9};
constexpr double kTsc0 = 7.0e14;      // host TSC at tau = 0
constexpr double kHostBase = 5.0e12;  // steady_clock ns at tau = 0
constexpr double kTicksPerNs = 3.0;   // the modelled TSC rate

double refclk(int chip, double tau) { return kDref[chip] + kRefHz * tau; }
double wall(int chip, double tau) {
    if (chip != 0 || tau <= kTauSwitch) {
        return kW0[chip] + kF0 * tau;
    }
    return kW0[chip] + kF0 * kTauSwitch + kSlow * kF0 * (tau - kTauSwitch);
}
// A 1588 stamp of the event at tau, in the link's stamp units (quarter-ns) of the refclk domain.
double hw_stamp(int chip, double tau) { return refclk(chip, tau) * 80.0; }
double tsc(double tau) { return kTsc0 + tau * 1e9 * kTicksPerNs; }
double host_ns(double tau) { return kHostBase + tau * 1e9; }

ClockSample sample(uint32_t dev, uint32_t lane, uint32_t kind, uint32_t round, uint32_t role, double rc, double w) {
    return ClockSample{
        dev, lane, kind, round, role, static_cast<uint64_t>(std::llround(rc)), static_cast<uint64_t>(std::llround(w))};
}

int main() {
    constexpr uint32_t kN = profiler::kSpscNRiscDecode;
    const CoreCoord e0{0, 11}, e1{1, 11};  // two distinct eth cores, for a chip that hosts two links

    // Per chip, the eth cores in its decode roster (the order fixes each core's index).
    const std::vector<std::vector<CoreCoord>> eth = {{e0}, {e0, e1}, {e0}};

    CaptureContext ctx;
    for (int c = 0; c < 3; c++) {
        CaptureContext::Device d;
        d.chip_id = static_cast<uint32_t>(c);
        for (const CoreCoord& ec : eth[c]) {
            for (uint32_t r = 0; r < kN; r++) {
                d.lanes.push_back(api::Core{
                    .logical = ec,
                    .physical = ec,
                    .chip_id = static_cast<uint32_t>(c),
                    .risc = static_cast<api::Risc>(r)});
            }
            d.core_xy.push_back(0);
        }
        d.n_eth_cores = static_cast<uint32_t>(eth[c].size());
        d.clock.chip_id = static_cast<uint32_t>(c);
        d.clock.frequency_ghz = kF0 * 1e-9;
        d.has_eth_tracker = true;
        ctx.devices.push_back(d);
    }
    ctx.links.push_back(CaptureContext::Link{
        .dev_a = 0, .dev_b = 1, .chip_a = 0, .chip_b = 1, .core_a = 0, .core_b = 0, .eth_a = e0, .eth_b = e0});
    ctx.links.push_back(CaptureContext::Link{
        .dev_a = 1, .dev_b = 2, .chip_a = 1, .chip_b = 2, .core_a = 1, .core_b = 0, .eth_a = e1, .eth_b = e0});
    ctx.root_dev = 0;
    SyncEngine sync;
    // The host series as the probe would write it: exact nodes at two bursts, so the checks cross a node and run
    // out along a tangent.
    for (double tau : {0.0, 0.6}) {
        sync.map().append_host(
            HostNode{.at = refclk(0, tau), .value = tsc(tau), .tangent = kTicksPerNs * 1e9 / kRefHz});
    }
    SteadySegment seg;
    seg.tsc0 = static_cast<int64_t>(kTsc0);
    seg.mono0 = static_cast<int64_t>(kHostBase);
    seg.ns_per_tick = 1.0 / kTicksPerNs;
    seg.ok = true;
    SteadyView::set(seg);

    sync.on_attach(ctx);
    // Trackers: the pushers' model points, one per ms on all three chips over one second, each behind 4095 samples,
    // except that chip 1 goes silent from 0.40 to 0.75 s: longer than the refclk's 24-bit period, so a stream
    // reassembled from its neighbours would come back a wrap off, and the check at 0.5 s lies inside the hole. Chip
    // 0's step: its first segment closes a microsecond before the switch, the next opens 50 us after it behind 64
    // samples and settles behind 1024 a hundred microseconds later.
    constexpr uint32_t kK8Fast = 216, kK8Slow = 215;  // 27.0 and 26.875 wall ticks per refclk tick, in eighths
    const auto point = [&](int c, double tau, uint32_t k8, uint32_t n, bool close) {
        sync.on_clock(sample(
            static_cast<uint32_t>(c),
            0,
            PP_CLOCK_LOCAL_REFCLK,
            k8 | (n << 8),
            close ? PP_CLOCK_LOCAL_CLOSE : PP_CLOCK_LOCAL_POINT,
            refclk(c, tau),
            wall(c, tau)));
    };
    bool switched = false;
    for (int k = 0; k < 1000; k++) {
        const double tau = k * 1e-3;
        if (!switched && tau > kTauSwitch) {
            point(0, kTauSwitch - 1e-6, kK8Fast, 4095, true);
            point(0, kTauSwitch + 50e-6, kK8Slow, 64, false);
            point(0, kTauSwitch + 150e-6, kK8Slow, 1024, false);
            switched = true;
        }
        for (int c = 0; c < 3; c++) {
            if (c == 1 && tau > 0.40 && tau < 0.75) {
                continue;
            }
            point(c, tau, c == 0 && tau > kTauSwitch ? kK8Slow : kK8Fast, 4095, false);
        }
    }
    // Two boot-time link bursts, 300 rounds each, 10 us apart. For (snd_dev, snd_lane) sender and (rcv_dev, rcv_lane)
    // receiver: sender stamps round start and end, receiver the arrival and its echo. The streams are damaged the way
    // a lapped consumer or a full ring damages them: the receiver's stamp is missing for every seventh round, the
    // sender's end stamp for every eleventh, and one receiver stamp arrives five rounds late.
    const auto burst = [&](uint32_t snd_dev, uint32_t snd_lane, uint32_t rcv_dev, uint32_t rcv_lane) {
        const auto receiver = [&](uint32_t k) {
            const double t = 0.020 + k * 10e-6;
            sync.on_clock(sample(
                rcv_dev,
                rcv_lane,
                PP_CLOCK_LINK_PTP,
                k,
                PP_CLOCK_ROLE_T1,
                hw_stamp(rcv_dev, t + kOneWay),
                wall(rcv_dev, t + kOneWay)));
            sync.on_clock(sample(
                rcv_dev,
                rcv_lane,
                PP_CLOCK_LINK_PTP,
                k,
                PP_CLOCK_ROLE_T1B,
                hw_stamp(rcv_dev, t + kOneWay + kTurn),
                wall(rcv_dev, t + kOneWay + kTurn)));
        };
        for (uint32_t k = 0; k < 300; k++) {
            const double t = 0.020 + k * 10e-6;
            sync.on_clock(sample(
                snd_dev, snd_lane, PP_CLOCK_LINK_PTP, k, PP_CLOCK_ROLE_T0, hw_stamp(snd_dev, t), wall(snd_dev, t)));
            if (k % 11 != 5) {
                sync.on_clock(sample(
                    snd_dev,
                    snd_lane,
                    PP_CLOCK_LINK_PTP,
                    k,
                    PP_CLOCK_ROLE_T2,
                    hw_stamp(snd_dev, t + 2 * kOneWay + kTurn),
                    wall(snd_dev, t + 2 * kOneWay + kTurn)));
            }
            if (k % 7 != 3 && k != 100) {
                receiver(k);
            }
            if (k == 105) {
                receiver(100);
            }
        }
    };
    burst(/*snd*/ 0, 0 * kN, /*rcv*/ 1, 0 * kN);  // link (0 e0 -> 1 e0): chip 1's e0 is core index 0
    burst(/*snd*/ 1, 1 * kN, /*rcv*/ 2, 0 * kN);  // link (1 e1 -> 2 e0): chip 1's e1 is core index 1
    sync.on_capture_end(ctx);

    // A record's placement: its eth wall tick through the chip's series, as the service places a record at release.
    const auto placed_ns = [&](int c, double tau) {
        const int64_t t = sync.map().lookup_tsc(static_cast<uint32_t>(c), std::llround(wall(c, tau)));
        return (static_cast<double>(t) - tsc(tau)) / kTicksPerNs;  // ns from the truth
    };
    const auto steady_ns = [&](int c, double tau) {
        return static_cast<double>(
            SteadyView::mono_ns(sync.map().lookup_tsc(static_cast<uint32_t>(c), std::llround(wall(c, tau)))));
    };
    char what[112];
    for (double tau : {0.050, 0.150, 0.280}) {
        std::snprintf(what, sizeof what, "(a) chip0 root pre-switch  tau=%.3f", tau);
        check_near(what, placed_ns(0, tau), 0.0, 5.0);
    }
    // (b): a node frozen on the wrong side of the switch would be off by 1/216 of the time since it, i.e. ms.
    for (double tau : {0.400, 0.700, 0.950}) {
        std::snprintf(what, sizeof what, "(b) chip0 root post-switch tau=%.3f", tau);
        check_near(what, placed_ns(0, tau), 0.0, 5.0);
    }
    for (double tau : {0.050, 0.500, 0.950}) {
        std::snprintf(what, sizeof what, "(c) chip1 one hop  tau=%.3f", tau);
        check_near(what, placed_ns(1, tau), 0.0, 5.0);
    }
    for (double tau : {0.050, 0.500, 0.950}) {
        std::snprintf(what, sizeof what, "(d) chip2 two hops tau=%.3f", tau);
        check_near(what, placed_ns(2, tau), 0.0, 5.0);
        std::snprintf(what, sizeof what, "(d) chip2 on the root refclk tau=%.3f", tau);
        check_near(
            what, (sync.map().lookup_root(2, std::llround(wall(2, tau))) - refclk(0, tau)) * (1e9 / kRefHz), 0.0, 5.0);
    }
    for (int c : {0, 1, 2}) {
        for (double tau : {0.050, 0.500, 0.950}) {
            std::snprintf(what, sizeof what, "steady chip%d tau=%.3f", c, tau);
            check_near(what, steady_ns(c, tau), host_ns(tau), 6.0);
        }
    }
    // The composed placement the service stamps records with must be the two-level lookup, everywhere: across chip
    // 0's step, across the host nodes, on the open tangents, on every chip.
    {
        double worst = 0.0;
        size_t n = 0;
        for (int c = 0; c < 3; c++) {
            for (double tau = 0.001; tau < 0.999; tau += 0.00037) {
                const int64_t w = std::llround(wall(c, tau));
                const int64_t two = sync.map().lookup_tsc(static_cast<uint32_t>(c), w);
                const int64_t one = sync.map().place_host(static_cast<uint32_t>(c), w);
                if (two == 0 || one == 0) {
                    continue;
                }
                const int64_t two_units = api::host_clock::from_tsc(two).time_since_epoch().count();
                worst = std::max(worst, std::fabs(static_cast<double>(one - two_units)) / 10.0);
                n++;
            }
        }
        // The reference is rounded to a whole TSC tick and the composed value to a host_clock unit.
        const double tick_ns = static_cast<double>(api::host_clock::from_tsc(1).time_since_epoch().count() -
                                                   api::host_clock::from_tsc(0).time_since_epoch().count()) /
                               10.0;
        std::snprintf(what, sizeof what, "composed placement vs two-level lookup over %zu instants (ns)", n);
        check_near(what, worst, 0.0, tick_ns + 0.1);
    }
    // A series past its capacity keeps its newest nodes: placement on them is unchanged, and a key before the oldest
    // kept node still places, on that node's tangent.
    {
        constexpr uint32_t chip = 3;
        constexpr int64_t step = 1000;
        const double a = 7.5e12, b = 0.037;  // root = a + b * wall, an exact line so every placement has one answer
        const uint32_t n = PlacementMap::kSeriesNodes + 1;
        for (uint32_t i = 0; i < n; i++) {
            const int64_t at = static_cast<int64_t>(i) * step;
            sync.map().append(chip, SyncNode{.at = at, .value = a + b * static_cast<double>(at), .tangent = b});
        }
        const auto on_line = [&](int64_t at) {
            return sync.map().lookup_root(chip, at) - (a + b * static_cast<double>(at));
        };
        check_near("retained: newest node", on_line(static_cast<int64_t>(n - 1) * step), 0.0, 1e-3);
        check_near("retained: a node mid-series", on_line(static_cast<int64_t>(n / 2) * step + step / 2), 0.0, 1e-3);
        check_near("retained: the retired first node (on the oldest kept tangent)", on_line(0), 0.0, 1e-3);
        if (sync.map().lookup_root(chip, 0) == 0.0) {
            std::printf("FAIL retained: a key before the oldest kept node no longer places\n");
            g_fail++;
        }
    }
    if (sync.map().lookup_tsc(2, std::llround(wall(2, 0.5))) == 0) {
        std::printf("FAIL (d) chip2 is not placed: the leaf never reached the root\n");
        g_fail++;
    }
    if (g_fail != 0) {
        std::printf("FAILED (%d)\n", g_fail);
        return 1;
    }
    std::printf("PASSED\n");
    return 0;
}
