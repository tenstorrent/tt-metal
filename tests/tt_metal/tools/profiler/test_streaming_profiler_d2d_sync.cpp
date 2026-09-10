// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host-only check of the d2d sync's time-indexed correction against a synthetic truth model, no device needed.
//
// Two chips whose refclks run at exactly 50 MHz with a known offset, whose AICLKs are known functions of true time
// (chip 0 slows by 1% mid-session: DVFS; chip 1 is steady), and whose boot anchors are what the device layer would
// measure (exact at the anchor instant, at the rate that applied then) -- except chip 1's host anchor, which is
// deliberately 50 us late. The records' own conversion (Record::host_time, reproduced here bit for bit with the baked
// hz/offset) plus the published correction must recover TRUE host time in every case:
//   (a) chip 0 before its switch: the anchor is right, the term is ~0;
//   (b) chip 0 after its switch: the base under-counts elapsed time by 1%, the local term restores it (root too);
//   (c) chip 1: only the link to the root can remove its anchor error, and it must.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "impl/streaming_profiler/spsc_packet.h"
#include "impl/streaming_profiler/streaming_profiler_d2d_sync.hpp"
#include "impl/streaming_profiler/streaming_profiler_sync_correction.hpp"

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

// ---- truth model ---------------------------------------------------------------------------------------------
constexpr double kRefHz = 50e6;
constexpr double kF0 = 1.35e9;             // AICLK at boot on both chips; the boot anchors measure exactly this rate
constexpr double kSlow = 0.99;             // chip 0's AICLK after its DVFS switch, relative to kF0
constexpr double kTauSwitch = 0.300;       // s, when chip 0 slows
constexpr double kD = 1.0e6;               // chip 1 refclk = chip 0 refclk + kD ticks (20 ms) at the same instant
constexpr double kOneWay = 1.0e-6;         // s, symmetric link one-way delay
constexpr double kAnchorErr1 = 50000.0;    // ns, how late chip 1's own host anchor is
constexpr double kW0[2] = {1.0e9, 7.0e9};  // wall tick origins at tau = 0
constexpr double kHostBase = 5.0e12;       // steady_clock ns at tau = 0

double refclk(int chip, double tau) { return (chip == 0 ? 0.0 : kD) + kRefHz * tau; }
double wall(int chip, double tau) {
    if (chip == 1 || tau <= kTauSwitch) {
        return kW0[chip] + kF0 * tau;
    }
    return kW0[chip] + kF0 * kTauSwitch + kSlow * kF0 * (tau - kTauSwitch);
}
double host_ns(double tau) { return kHostBase + tau * 1e9; }

ClockSample sample(uint32_t dev, uint32_t kind, double rc, double w) {
    return ClockSample{
        dev,
        /*lane=*/0,
        kind,
        static_cast<uint32_t>(static_cast<uint64_t>(std::llround(rc)) & 0xFFFFFFu),
        static_cast<uint64_t>(std::llround(w))};
}

int main() {
    constexpr uint32_t kN = profiler::kSpscNRiscDecode;
    const CoreCoord eth{0, 11};
    const double tau_anchor[2] = {0.010, 0.012};  // boot anchors 10 and 12 ms after the trackers start

    CaptureContext ctx;
    for (int c = 0; c < 2; c++) {
        CaptureContext::Device d;
        d.chip_id = static_cast<uint32_t>(c);
        for (uint32_t r = 0; r < kN; r++) {
            d.lanes.push_back(api::Core{
                .logical = eth,
                .physical = eth,
                .chip_id = static_cast<uint32_t>(c),
                .risc = static_cast<api::Risc>(r)});
        }
        d.core_xy.push_back(0);
        d.clock.chip_id = static_cast<uint32_t>(c);
        d.clock.frequency_ghz = kF0 * 1e-9;
        d.clock.anchor_ticks = static_cast<uint64_t>(std::llround(wall(c, tau_anchor[c])));
        d.clock.anchor_host_ns =
            static_cast<int64_t>(std::llround(host_ns(tau_anchor[c]) + (c == 1 ? kAnchorErr1 : 0.0)));
        ctx.devices.push_back(d);
    }
    ctx.links.push_back(
        CaptureContext::Link{.dev_a = 0, .dev_b = 1, .chip_a = 0, .chip_b = 1, .eth_a = eth, .eth_b = eth});

    D2dSyncConsumer sync;
    sync.on_attach(ctx);
    // Trackers: a LOCAL sample every 3 us on both chips over one second; the 24-bit refclk payload wraps ~3 times.
    for (double tau = 0.0; tau < 1.0; tau += 3e-6) {
        for (int c = 0; c < 2; c++) {
            sync.on_clock(sample(static_cast<uint32_t>(c), PP_CLOCK_LOCAL_REFCLK, refclk(c, tau), wall(c, tau)));
        }
    }
    // The boot-time link burst: 240 rounds from tau = 20 ms, 10 us apart. Sender (chip 0) stamps start and end,
    // receiver (chip 1) the arrival, in the order the kernels emit them.
    for (int k = 0; k < 240; k++) {
        const double t = 0.020 + k * 10e-6;
        sync.on_clock(sample(0, PP_CLOCK_LINK_REFCLK, refclk(0, t), wall(0, t)));
        sync.on_clock(sample(0, PP_CLOCK_LINK_REFCLK, refclk(0, t + 2 * kOneWay), wall(0, t + 2 * kOneWay)));
        sync.on_clock(sample(1, PP_CLOCK_LINK_REFCLK, refclk(1, t + kOneWay), wall(1, t + kOneWay)));
    }
    sync.on_capture_end(ctx);
    std::printf(
        "segments published: chip0 %zu chip1 %zu\n", SyncCorrections::published(0), SyncCorrections::published(1));

    // Record::host_time, reproduced: the baked hz/offset exactly as record_consts bakes them, plus the term.
    const auto record_host_ns = [&](int c, double T) {
        const DeviceClock& k = ctx.devices[c].clock;
        const uint32_t hz = static_cast<uint32_t>(std::llround(k.frequency_ghz * 1e9));
        const int64_t offset =
            std::llround(static_cast<double>(k.anchor_host_ns) * (hz * 1e-9)) - static_cast<int64_t>(k.anchor_ticks);
        const uint64_t ticks = static_cast<uint64_t>(std::llround(T));
        const double cycles = static_cast<double>(static_cast<int64_t>(ticks) + offset);
        const int64_t base = static_cast<int64_t>(cycles * 1e9 / hz);
        return static_cast<double>(base + SyncCorrections::lookup_ns(static_cast<uint32_t>(c), ticks));
    };
    char what[96];
    // (a)
    for (double tau : {0.050, 0.150, 0.280}) {
        std::snprintf(what, sizeof what, "(a) chip0 pre-switch  tau=%.3f", tau);
        check_near(what, record_host_ns(0, wall(0, tau)), host_ns(tau), 200.0);
    }
    // (b): without the term the error here would be 0.01 * (tau - 0.3) s, i.e. 1 to 6.5 ms.
    for (double tau : {0.400, 0.700, 0.950}) {
        std::snprintf(what, sizeof what, "(b) chip0 post-switch tau=%.3f", tau);
        check_near(what, record_host_ns(0, wall(0, tau)), host_ns(tau), 2000.0);
    }
    // (c): without the link the error here would be the 50 us anchor error.
    for (double tau : {0.050, 0.500, 0.950}) {
        std::snprintf(what, sizeof what, "(c) chip1 on root timeline tau=%.3f", tau);
        check_near(what, record_host_ns(1, wall(1, tau)), host_ns(tau), 200.0);
    }
    check_near(
        "(c) chip1 term ~ -anchor error",
        static_cast<double>(SyncCorrections::lookup_ns(1, static_cast<uint64_t>(std::llround(wall(1, 0.5))))),
        -kAnchorErr1,
        200.0);
    if (g_fail != 0) {
        std::printf("FAILED (%d)\n", g_fail);
        return 1;
    }
    std::printf("PASSED\n");
    return 0;
}
