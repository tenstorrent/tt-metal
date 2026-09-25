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
// (sync.map().place_host on the record's eth wall tick) must recover TRUE host time in every case:
//   (a) chip 0 before its switch;
//   (b) chip 0 after its switch: the run boundary must be placed where the two lines meet;
//   (c) chip 1 (one hop): only the 0-1 link places it;
//   (d) chip 2 (two hops): only 0-1 composed with 1-2 places it, refclk offset and all.
// The root-refclk placement (the d2d level, which the host series never enters) and the steady_clock view
// (steady_mono_ns through a known steady series) are checked alongside.
//
// Chip 1 is a receiver (of 0-1) AND a sender (of 1-2), on two DIFFERENT eth cores -- exactly as real hardware, where
// each link owns its own eth core -- so its two stamp streams stay separate.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "tt_metal/common/indexed_ring.hpp"
#include "impl/streaming_profiler/streaming_profiler_service.hpp"
#include "impl/streaming_profiler/streaming_profiler_sync_engine.hpp"
#include "impl/streaming_profiler/streaming_profiler_sync_devices.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_metal::streaming_profiler;
namespace api = tt::tt_metal::experimental::streaming_profiler;

// The sync's record kinds and roles (hostdev/streaming_profiler_sync.h).
constexpr uint32_t kLocal = kernel_profiler::kSyncKindLocal, kLink = kernel_profiler::kSyncKindLink;
constexpr uint32_t kT0 = kernel_profiler::kSyncRoleT0, kT1 = kernel_profiler::kSyncRoleT1;
constexpr uint32_t kT1B = kernel_profiler::kSyncRoleT1B, kT2 = kernel_profiler::kSyncRoleT2;

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
// The test's readings are exact, so a placement is off only by its own rounding: a unit of host_clock (0.1 ns) and a
// TSC tick (0.33 ns) on the way back.
constexpr double kTol = 0.5;
constexpr double kF0 = 1.35e9;           // AICLK at boot on every chip
constexpr double kSlow = 26.875 / 27.0;  // chip 0's AICLK after its DVFS switch: one 1/8 step of the PLL multiple
constexpr double kTauSwitch = 0.300;     // s, when chip 0 slows
constexpr double kOneWay = 1.0e-6;       // s, symmetric link one-way delay
constexpr double kTurn = 350e-9;  // s, the receiver's turnaround: its echo leaves this long after the frame arrived
// Every clock reads as it would a year after power-on, the host's TSC and steady_clock a year after boot, so each count
// is far past 2^53 of its units. Chip c's refclk reads kRef0[c] at tau = 0 (their differences are the offsets the
// links must recover) and its wall kWall0[c].
constexpr int64_t kYear = int64_t{365} * 86400;
constexpr int64_t kRef0[3] = {kYear * 50'000'000, kYear * 50'000'000 + 1'000'000, kYear * 50'000'000 + 3'000'000};
constexpr int64_t kWall0[3] = {
    kYear * 1'350'000'000 + 1'000'000'000,
    kYear * 1'350'000'000 + 7'000'000'000,
    kYear * 1'350'000'000 + 4'000'000'000};
constexpr int64_t kTsc0 = kYear * 3'000'000'000;   // host TSC at tau = 0
constexpr int64_t kHost0 = kYear * 1'000'000'000;  // steady_clock ns at tau = 0
constexpr double kTicksPerNs = 3.0;                // the modelled TSC rate

// Each clock at tau from its reading at tau = 0; every reading the test takes is a whole number of its units.
double refclk(double tau) { return kRefHz * tau; }
double wall(int chip, double tau) {
    if (chip != 0 || tau <= kTauSwitch) {
        return kF0 * tau;
    }
    return kF0 * kTauSwitch + kSlow * kF0 * (tau - kTauSwitch);
}
double tsc(double tau) { return tau * 1e9 * kTicksPerNs; }
double host_ns(double tau) { return tau * 1e9; }
int64_t wall_tick(int chip, double tau) { return kWall0[chip] + std::llround(wall(chip, tau)); }
// A 1588 stamp of the event at tau, in the link's stamp units (kLinkSyncStampUnitsPerNs per ns) of the refclk domain.
uint64_t hw_stamp(int chip, double tau) {
    constexpr int64_t kUnitsPerRefclk =
        int64_t{kernel_profiler::kLinkSyncStampUnitsPerNs} * (1'000'000'000 / kernel_profiler::kEthRefclkHz);
    return static_cast<uint64_t>(kRef0[chip] * kUnitsPerRefclk + std::llround(refclk(tau) * kUnitsPerRefclk));
}

// Hands the engine one sync record as the stream carries it.
void feed(
    SyncEngine& sync,
    uint32_t dev,
    uint32_t core,
    uint32_t kind,
    uint32_t round,
    uint32_t role,
    uint64_t value,
    uint64_t wall) {
    const kernel_profiler::SyncRecord rec{
        .meta = kernel_profiler::word_of(kernel_profiler::SyncMeta{.role = role, .kind = kind}),
        .round = round,
        .value_lo = static_cast<uint32_t>(value),
        .value_hi = static_cast<uint32_t>(value >> 32),
        .wall_lo = static_cast<uint32_t>(wall),
        .wall_hi = static_cast<uint32_t>(wall >> 32)};
    sync.on_clock(dev, core, rec);
}

int main() {
    CaptureContext ctx;
    for (uint32_t c = 0; c < 3; c++) {
        ctx.devices.push_back({.chip_id = c});
    }
    ctx.links.push_back(
        CaptureContext::Link{.dev_a = 0, .dev_b = 1, .chip_a = 0, .chip_b = 1, .core_a = 0, .core_b = 0});
    ctx.links.push_back(
        CaptureContext::Link{.dev_a = 1, .dev_b = 2, .chip_a = 1, .chip_b = 2, .core_a = 1, .core_b = 0});
    ctx.root_dev = 0;
    SyncEngine sync(4 * tt::tt_metal::IndexedRing<SyncNode>::kChunkItems);  // small enough to wrap below
    // The host series as the probe would write it: exact nodes at two bursts, so the checks cross a node and run
    // out along a tangent.
    sync.map().set_bases(kRef0[0], kTsc0);
    for (double tau : {0.0, 0.6}) {
        sync.map().append_host(HostNode{.at = refclk(tau), .value = tsc(tau), .tangent = kTicksPerNs * 1e9 / kRefHz});
    }
    // The steady series as the probe would write it, in the process's clock map, which steady_mono_ns reads.
    for (double tau : {0.0, 1.0}) {
        service().sync().map().append_steady(
            kTsc0 + std::llround(tsc(tau)), kHost0 + std::llround(host_ns(tau)), 1.0 / kTicksPerNs);
    }

    sync.on_attach(ctx);
    // Trackers: the pushers' model points, one per ms on all three chips over one second, one point to a LOCAL record,
    // except that chip 1 goes silent from 0.40 to 0.75 s: longer than the refclk's 24-bit period, so a stream
    // reassembled from its neighbours would come back a wrap off, and the check at 0.5 s lies inside the hole. Chip
    // 0's step: its first segment closes a microsecond before the switch, the next opens 50 us after it and has a
    // second point a hundred microseconds later.
    constexpr uint32_t kK8Fast = 216, kK8Slow = 215;  // 27.0 and 26.875 wall ticks per refclk tick, in eighths
    const auto point = [&](int c, double tau, uint32_t k8, bool close) {
        feed(
            sync,
            static_cast<uint32_t>(c),
            0,
            kLocal,
            k8 | (k8 << 24),
            1u | (close ? 1u << 2 : 0u),
            static_cast<uint64_t>(kRef0[c] + std::llround(refclk(tau))),
            static_cast<uint64_t>(8 * kWall0[c] + std::llround(8.0 * wall(c, tau))));  // a point's wall is in eighths
    };
    bool switched = false;
    for (int k = 0; k < 1000; k++) {
        const double tau = k * 1e-3;
        if (!switched && tau > kTauSwitch) {
            point(0, kTauSwitch - 1e-6, kK8Fast, true);
            point(0, kTauSwitch + 50e-6, kK8Slow, false);
            point(0, kTauSwitch + 150e-6, kK8Slow, false);
            switched = true;
        }
        for (int c = 0; c < 3; c++) {
            if (c == 1 && tau > 0.40 && tau < 0.75) {
                continue;
            }
            point(c, tau, c == 0 && tau > kTauSwitch ? kK8Slow : kK8Fast, false);
        }
    }
    // Two boot-time link bursts, 300 rounds each, 10 us apart. For (snd_dev, snd_core) sender and (rcv_dev, rcv_core)
    // receiver: each end records the peer's egress stamp, read from the frame, and its own ingress stamp -- the
    // receiver the frame's (T0, T1), the sender the echo's (T1B, T2). The streams are damaged the way a lapped
    // consumer or a full ring damages them: the receiver's stamps are missing for every seventh round, the sender's
    // ingress stamp for every eleventh, and one round of the receiver's arrives five rounds late.
    const auto burst = [&](uint32_t snd_dev, uint32_t snd_core, uint32_t rcv_dev, uint32_t rcv_core) {
        const auto receiver = [&](uint32_t k) {
            const double t = 0.020 + k * 10e-6;
            const auto w = static_cast<uint64_t>(wall_tick(rcv_dev, t + kOneWay));
            feed(sync, rcv_dev, rcv_core, kLink, k, kT0, hw_stamp(snd_dev, t), w);
            feed(sync, rcv_dev, rcv_core, kLink, k, kT1, hw_stamp(rcv_dev, t + kOneWay), w);
        };
        for (uint32_t k = 0; k < 300; k++) {
            const double t = 0.020 + k * 10e-6;
            const double echo_out = t + kOneWay + kTurn, echo_in = t + 2 * kOneWay + kTurn;
            const auto w = static_cast<uint64_t>(wall_tick(snd_dev, echo_in));
            feed(sync, snd_dev, snd_core, kLink, k, kT1B, hw_stamp(rcv_dev, echo_out), w);
            if (k % 11 != 5) {
                feed(sync, snd_dev, snd_core, kLink, k, kT2, hw_stamp(snd_dev, echo_in), w);
            }
            if (k % 7 != 3 && k != 100) {
                receiver(k);
            }
            if (k == 105) {
                receiver(100);
            }
        }
    };
    burst(/*snd*/ 0, 0, /*rcv*/ 1, 0);
    burst(/*snd*/ 1, 1, /*rcv*/ 2, 0);
    sync.on_capture_end();

    // A record's placement: its eth wall tick through the chip's series, as the service places a record at release,
    // read back as the TSC tick it is on host_clock.
    const auto placed_tsc = [&](int c, double tau) {
        const int64_t units = sync.map().place_host(static_cast<uint32_t>(c), wall_tick(c, tau));
        return api::host_clock::tsc(api::host_clock::time_point(api::host_clock::duration(units)));
    };
    const auto placed_ns = [&](int c, double tau) {
        return (static_cast<double>(placed_tsc(c, tau) - kTsc0) - tsc(tau)) / kTicksPerNs;  // ns from the truth
    };
    const auto steady_ns = [&](int c, double tau) {
        return static_cast<double>(steady_mono_ns(placed_tsc(c, tau)) - kHost0);
    };
    char what[112];
    for (double tau : {0.050, 0.150, 0.280}) {
        std::snprintf(what, sizeof what, "(a) chip0 root pre-switch  tau=%.3f", tau);
        check_near(what, placed_ns(0, tau), 0.0, kTol);
    }
    // (b): a node frozen on the wrong side of the switch would be off by 1/216 of the time since it, i.e. ms.
    for (double tau : {0.400, 0.700, 0.950}) {
        std::snprintf(what, sizeof what, "(b) chip0 root post-switch tau=%.3f", tau);
        check_near(what, placed_ns(0, tau), 0.0, kTol);
    }
    for (double tau : {0.050, 0.500, 0.950}) {
        std::snprintf(what, sizeof what, "(c) chip1 one hop  tau=%.3f", tau);
        check_near(what, placed_ns(1, tau), 0.0, kTol);
    }
    for (double tau : {0.050, 0.500, 0.950}) {
        std::snprintf(what, sizeof what, "(d) chip2 two hops tau=%.3f", tau);
        check_near(what, placed_ns(2, tau), 0.0, kTol);
        std::snprintf(what, sizeof what, "(d) chip2 on the root refclk tau=%.3f", tau);
        check_near(
            what,
            (sync.map().lookup_root(2, wall_tick(2, tau)).value_or(0.0) - refclk(tau)) * (1e9 / kRefHz),
            0.0,
            kTol);
    }
    for (int c : {0, 1, 2}) {
        for (double tau : {0.050, 0.500, 0.950}) {
            std::snprintf(what, sizeof what, "steady chip%d tau=%.3f", c, tau);
            check_near(what, steady_ns(c, tau), host_ns(tau), 1.0);
        }
    }
    // A series past its capacity keeps its newest nodes: placement on them is unchanged, and a key before the oldest
    // kept node still places, on that node's tangent.
    {
        constexpr uint32_t chip = 3;
        constexpr int64_t step = 1000;
        const double a = 7.5e12, b = 0.037;  // root = a + b * wall, an exact line so every placement has one answer
        const uint32_t n = sync.map().series_nodes() + 1;
        for (uint32_t i = 0; i < n; i++) {
            const int64_t at = static_cast<int64_t>(i) * step;
            sync.map().append(chip, SyncNode{.at = at, .value = a + b * static_cast<double>(at), .tangent = b});
        }
        const auto on_line = [&](int64_t at) {
            return sync.map().lookup_root(chip, at).value_or(0.0) - (a + b * static_cast<double>(at));
        };
        check_near("retained: newest node", on_line(static_cast<int64_t>(n - 1) * step), 0.0, 1e-3);
        check_near("retained: a node mid-series", on_line(static_cast<int64_t>(n / 2) * step + step / 2), 0.0, 1e-3);
        check_near("retained: the retired first node (on the oldest kept tangent)", on_line(0), 0.0, 1e-3);
    }
    if (g_fail != 0) {
        std::printf("FAILED (%d)\n", g_fail);
        return 1;
    }
    std::printf("PASSED\n");
    return 0;
}
