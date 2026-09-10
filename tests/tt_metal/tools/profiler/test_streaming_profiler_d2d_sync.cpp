// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host-only check of the d2d sync's time-indexed correction against a synthetic truth model, no device needed.
//
// A three-chip CHAIN: root(0) -- mid(1) -- leaf(2), with solved links only for (0,1) and (1,2). The leaf has NO
// direct link to the root, so it can only reach the fleet timeline by composing the two hops -- that composition is
// what this test exercises beyond the single-link case. The chips' refclks run at exactly 50 MHz with known offsets;
// their AICLKs are known functions of true time (chip 0 slows 1% mid-session: DVFS; chips 1 and 2 steady); and their
// boot anchors are what the device layer would measure (exact at the anchor, at the rate that applied then) except
// that chips 1 and 2 have deliberately-late host anchors. The record's own conversion (Record::host_time, reproduced
// bit for bit with the baked hz/offset) plus the published correction must recover TRUE host time in every case:
//   (a) chip 0 before its switch: the anchor is right, the term is ~0;
//   (b) chip 0 after its switch: the base under-counts by 1%, the local term restores it (root too);
//   (c) chip 1 (one hop): only the 0-1 link removes its anchor error;
//   (d) chip 2 (two hops): only 0-1 composed with 1-2 removes its anchor error and its refclk offset.
//
// Chip 1 is a receiver (of 0-1) AND a sender (of 1-2), on two DIFFERENT eth cores -- exactly as real hardware, where
// each link owns its own eth core -- so its two stamp streams stay separate.

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
constexpr double kF0 = 1.35e9;        // AICLK at boot on every chip; the boot anchors measure exactly this rate
constexpr double kSlow = 0.99;        // chip 0's AICLK after its DVFS switch
constexpr double kTauSwitch = 0.300;  // s, when chip 0 slows
constexpr double kOneWay = 1.0e-6;    // s, symmetric link one-way delay
// chip c refclk = kDref[c] + 50 MHz * tau (the offsets the links must recover); wall origin kW0[c].
constexpr double kDref[3] = {0.0, 1.0e6, 3.0e6};
constexpr double kW0[3] = {1.0e9, 7.0e9, 4.0e9};
constexpr double kAnchorErr[3] = {0.0, 50000.0, 30000.0};  // ns, how late each chip's own host anchor is
constexpr double kHostBase = 5.0e12;                       // steady_clock ns at tau = 0

double refclk(int chip, double tau) { return kDref[chip] + kRefHz * tau; }
double wall(int chip, double tau) {
    if (chip != 0 || tau <= kTauSwitch) {
        return kW0[chip] + kF0 * tau;
    }
    return kW0[chip] + kF0 * kTauSwitch + kSlow * kF0 * (tau - kTauSwitch);
}
double host_ns(double tau) { return kHostBase + tau * 1e9; }

ClockSample sample(uint32_t dev, uint32_t lane, uint32_t kind, double rc, double w) {
    return ClockSample{
        dev,
        lane,
        kind,
        static_cast<uint32_t>(static_cast<uint64_t>(std::llround(rc)) & 0xFFFFFFu),
        static_cast<uint64_t>(std::llround(w))};
}

int main() {
    constexpr uint32_t kN = profiler::kSpscNRiscDecode;
    const CoreCoord e0{0, 11}, e1{1, 11};  // two distinct eth cores, for a chip that hosts two links
    const double tau_anchor[3] = {0.010, 0.012, 0.014};

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
        d.clock.chip_id = static_cast<uint32_t>(c);
        d.clock.frequency_ghz = kF0 * 1e-9;
        d.clock.anchor_ticks = static_cast<uint64_t>(std::llround(wall(c, tau_anchor[c])));
        d.clock.anchor_host_ns = static_cast<int64_t>(std::llround(host_ns(tau_anchor[c]) + kAnchorErr[c]));
        ctx.devices.push_back(d);
    }
    // Links: (0 e0 -> 1 e0) and (1 e1 -> 2 e0). Chip 1 receives on e0, sends on e1.
    ctx.links.push_back(
        CaptureContext::Link{.dev_a = 0, .dev_b = 1, .chip_a = 0, .chip_b = 1, .eth_a = e0, .eth_b = e0});
    ctx.links.push_back(
        CaptureContext::Link{.dev_a = 1, .dev_b = 2, .chip_a = 1, .chip_b = 2, .eth_a = e1, .eth_b = e0});

    D2dSyncConsumer sync;
    sync.on_attach(ctx);
    // Trackers: a LOCAL sample every 3 us on all three chips over one second; the 24-bit refclk payload wraps ~3x.
    for (double tau = 0.0; tau < 1.0; tau += 3e-6) {
        for (int c = 0; c < 3; c++) {
            sync.on_clock(sample(static_cast<uint32_t>(c), 0, PP_CLOCK_LOCAL_REFCLK, refclk(c, tau), wall(c, tau)));
        }
    }
    // Two boot-time link bursts, 240 rounds each, 10 us apart. For (snd_dev, snd_lane) sender and (rcv_dev, rcv_lane)
    // receiver: sender stamps round start and end, receiver the arrival, in the order the kernels emit them.
    const auto burst = [&](uint32_t snd_dev, uint32_t snd_lane, uint32_t rcv_dev, uint32_t rcv_lane) {
        for (int k = 0; k < 240; k++) {
            const double t = 0.020 + k * 10e-6;
            sync.on_clock(sample(snd_dev, snd_lane, PP_CLOCK_LINK_REFCLK, refclk(snd_dev, t), wall(snd_dev, t)));
            sync.on_clock(sample(
                snd_dev,
                snd_lane,
                PP_CLOCK_LINK_REFCLK,
                refclk(snd_dev, t + 2 * kOneWay),
                wall(snd_dev, t + 2 * kOneWay)));
            sync.on_clock(sample(
                rcv_dev, rcv_lane, PP_CLOCK_LINK_REFCLK, refclk(rcv_dev, t + kOneWay), wall(rcv_dev, t + kOneWay)));
        }
    };
    burst(/*snd*/ 0, 0 * kN, /*rcv*/ 1, 0 * kN);  // link (0 e0 -> 1 e0): chip 1's e0 is core index 0
    burst(/*snd*/ 1, 1 * kN, /*rcv*/ 2, 0 * kN);  // link (1 e1 -> 2 e0): chip 1's e1 is core index 1
    sync.on_capture_end(ctx);
    std::printf(
        "segments published: chip0 %zu chip1 %zu chip2 %zu\n",
        SyncCorrections::published(0),
        SyncCorrections::published(1),
        SyncCorrections::published(2));

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
    char what[112];
    for (double tau : {0.050, 0.150, 0.280}) {
        std::snprintf(what, sizeof what, "(a) chip0 root pre-switch  tau=%.3f", tau);
        check_near(what, record_host_ns(0, wall(0, tau)), host_ns(tau), 200.0);
    }
    // (b): without the term the error here would be 0.01 * (tau - 0.3) s, i.e. 1 to 6.5 ms.
    for (double tau : {0.400, 0.700, 0.950}) {
        std::snprintf(what, sizeof what, "(b) chip0 root post-switch tau=%.3f", tau);
        check_near(what, record_host_ns(0, wall(0, tau)), host_ns(tau), 2000.0);
    }
    // (c) chip 1, one hop: without the link the error would be its 50 us anchor error.
    for (double tau : {0.050, 0.500, 0.950}) {
        std::snprintf(what, sizeof what, "(c) chip1 one hop  tau=%.3f", tau);
        check_near(what, record_host_ns(1, wall(1, tau)), host_ns(tau), 300.0);
    }
    // (d) chip 2, TWO hops (no direct link to root): without the composition the error would be its 30 us anchor
    // error, and a single-hop-only implementation would leave it uncorrected entirely.
    for (double tau : {0.050, 0.500, 0.950}) {
        std::snprintf(what, sizeof what, "(d) chip2 two hops tau=%.3f", tau);
        check_near(what, record_host_ns(2, wall(2, tau)), host_ns(tau), 400.0);
    }
    check_near(
        "(d) chip2 term ~ -anchor error (proves 2-hop composition ran)",
        static_cast<double>(SyncCorrections::lookup_ns(2, static_cast<uint64_t>(std::llround(wall(2, 0.5))))),
        -kAnchorErr[2],
        400.0);
    if (SyncCorrections::published(2) == 0) {
        std::printf("FAIL (d) chip2 published 0 segments: the leaf never reached the root timeline\n");
        g_fail++;
    }
    if (g_fail != 0) {
        std::printf("FAILED (%d)\n", g_fail);
        return 1;
    }
    std::printf("PASSED\n");
    return 0;
}
