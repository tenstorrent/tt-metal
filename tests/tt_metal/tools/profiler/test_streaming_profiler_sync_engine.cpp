// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host-only: the sync engine and clock map against a synthetic truth model, with no device. Chain feeds three
// chips joined in a chain by two links, with chips 1 and 2 on crystals tens of ppm off chip 0's, chip 0 switching AICLK
// partway, chip 1 silent for longer than the solve's 250 ms link window, link streams with dropped and late stamps, and
// every counter a year past power-on. It checks placements on the host timeline before, across and after each of those
// to within 0.5 ns, and steady_clock times to within 1 ns. Retention overflows a series' capacity and checks where
// records before and between the kept nodes, and on the newest, land.

#include <cmath>
#include <cstdint>
#include <optional>

#include <gtest/gtest.h>

#include "impl/streaming_profiler/service.hpp"
#include "impl/streaming_profiler/sync/engine.hpp"
#include "impl/streaming_profiler/sync/clock_map.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_metal::streaming_profiler;

namespace {

using kernel_profiler::SyncRole;

constexpr double kRefHz = kernel_profiler::kEthRefclkHz;
// The test's readings are exact, so a placement is off only by its rounding to a TSC tick (0.33 ns).
constexpr double kTolNs = 0.5;
// Wall ticks per refclk tick, in eighths, before and after chip 0's DVFS switch: one 1/8 step of the PLL multiple.
constexpr uint32_t kRateFast = 216, kRateSlow = 215;
constexpr double kAiclkHz = kRefHz * kRateFast / 8;
constexpr double kSlowAiclkRatio = static_cast<double>(kRateSlow) / kRateFast;
constexpr double kSwitchS = 0.300;
constexpr double kOneWayS = 1.0e-6;
constexpr double kTurnaroundS = 350e-9;
// Every clock reads as it would a year after power-on: the AICLK, TSC and host counts are past 2^53 of their units.
constexpr int64_t kYear = int64_t{365} * 86400;
constexpr int64_t kRef0[3] = {kYear * 50'000'000, kYear * 50'000'000 + 1'000'000, kYear * 50'000'000 + 3'000'000};
constexpr int64_t kWall0[3] = {
    kYear * 1'350'000'000 + 1'000'000'000,
    kYear * 1'350'000'000 + 7'000'000'000,
    kYear * 1'350'000'000 + 4'000'000'000};
constexpr int64_t kTsc0 = kYear * 3'000'000'000;
constexpr int64_t kHost0 = kYear * 1'000'000'000;
constexpr double kTicksPerNs = 3.0;
// Each chip's refclk and AICLK come from its own crystal, so a chip's rate offset scales both. A solver that drops a
// link's slope or picks the wrong window misplaces chips 1 and 2 by microseconds over the second.
constexpr double kRate[3] = {1.0, 1.0 + 40e-6, 1.0 - 25e-6};

double refclk(int chip, double tau) { return kRefHz * kRate[chip] * tau; }
double wall(int chip, double tau) {
    if (chip != 0 || tau <= kSwitchS) {
        return kRate[chip] * kAiclkHz * tau;
    }
    return kAiclkHz * kSwitchS + kSlowAiclkRatio * kAiclkHz * (tau - kSwitchS);
}
double tsc(double tau) { return tau * 1e9 * kTicksPerNs; }
double host_ns(double tau) { return tau * 1e9; }
int64_t wall_tick(int chip, double tau) { return kWall0[chip] + std::llround(wall(chip, tau)); }
uint64_t hw_stamp(int chip, double tau) {
    return static_cast<uint64_t>(kRef0[chip] * kNsPerRefclk + std::llround(refclk(chip, tau) * kNsPerRefclk));
}

void feed_local(SyncEngine& sync, uint32_t dev, uint64_t refclk, uint64_t wall8, uint32_t wall_per_refclk_eighths) {
    sync.on_record(
        dev,
        0,
        {.local = {
             .meta = kernel_profiler::word_of(
                 kernel_profiler::SyncMeta{.count = 1, .kind = kernel_profiler::SyncKind::Local}),
             .rates = kernel_profiler::word_of(kernel_profiler::SyncLocalRates{
                 .wall_per_refclk_eighths = {static_cast<uint8_t>(wall_per_refclk_eighths)},
                 .base_wall_per_refclk_eighths = static_cast<uint8_t>(wall_per_refclk_eighths)}),
             .first_refclk = refclk,
             .first_wall8 = wall8}});
}
void feed_link(SyncEngine& sync, uint32_t dev, uint32_t core, uint32_t round, SyncRole role, uint64_t stamp) {
    sync.on_record(
        dev,
        core,
        {.link = {
             .meta = kernel_profiler::word_of(
                 kernel_profiler::SyncMeta{.role = role, .kind = kernel_profiler::SyncKind::Link}),
             .round = round,
             .first = stamp,
             .count = 1}});
}

}  // namespace

TEST(StreamingProfilerSyncEngine, Chain) {
    CaptureContext ctx;
    for (uint32_t chip = 0; chip < 3; chip++) {
        ctx.devices.push_back({.chip_id = chip});
    }
    ctx.links.push_back(CaptureContext::Link{.dev_a = 0, .dev_b = 1, .core_a = 0, .core_b = 0});
    ctx.links.push_back(CaptureContext::Link{.dev_a = 1, .dev_b = 2, .core_a = 1, .core_b = 0});
    ClockMap map(3, ClockMap::kSeriesNodes, {.root_refclk = kRef0[0], .tsc = kTsc0});
    for (double tau : {0.0, 0.6}) {
        map.append_host(
            HostNode{.at = refclk(0, tau), .value = tsc(tau), .tangent = kTicksPerNs * 1e9 / kRefHz},
            refclk(0, tau + 0.6));
    }
    for (double tau : {0.0, 1.0}) {
        service().steady().append(kTsc0 + std::llround(tsc(tau)), kHost0 + std::llround(host_ns(tau)));
    }

    SyncEngine sync(ctx, map);
    // Chip 1 goes silent from 0.40 to 0.75 s, longer than the solve's 250 ms window.
    // A point is taken as the refclk reaches a whole tick, at or after tau.
    const auto point = [&](int chip, double tau, uint32_t wall_per_refclk_eighths) {
        const double ticks = std::ceil(refclk(chip, tau));
        const double at = ticks / (kRefHz * kRate[chip]);
        feed_local(
            sync,
            static_cast<uint32_t>(chip),
            static_cast<uint64_t>(kRef0[chip] + static_cast<int64_t>(ticks)),
            static_cast<uint64_t>(8 * kWall0[chip] + std::llround(8.0 * wall(chip, at))),
            wall_per_refclk_eighths);
    };
    bool switched = false;
    for (int k = 0; k < 1000; k++) {
        const double tau = k * 1e-3;
        if (!switched && tau > kSwitchS) {
            point(0, kSwitchS - 1e-6, kRateFast);
            point(0, kSwitchS + 50e-6, kRateSlow);
            point(0, kSwitchS + 150e-6, kRateSlow);
            switched = true;
        }
        for (int c = 0; c < 3; c++) {
            if (c == 1 && tau > 0.40 && tau < 0.75) {
                continue;
            }
            point(c, tau, c == 0 && tau > kSwitchS ? kRateSlow : kRateFast);
        }
    }
    // Each link runs 970 rounds 1 ms apart across the second the chips' points cover, several of the solve's 250 ms
    // windows, with its stream damaged the way a lapped consumer or a full ring damages it.
    constexpr uint32_t kLinkRounds = 970, kLateRound = 100, kLateArrival = 105;
    const auto burst = [&](uint32_t snd_dev, uint32_t snd_core, uint32_t rcv_dev, uint32_t rcv_core) {
        const auto receiver = [&](uint32_t round) {
            const double sent = 0.020 + round * 1e-3;
            feed_link(sync, rcv_dev, rcv_core, round, SyncRole::ForwardEgress, hw_stamp(snd_dev, sent));
            feed_link(sync, rcv_dev, rcv_core, round, SyncRole::ForwardIngress, hw_stamp(rcv_dev, sent + kOneWayS));
        };
        for (uint32_t k = 0; k < kLinkRounds; k++) {
            const double sent = 0.020 + k * 1e-3;
            const double echo_out = sent + kOneWayS + kTurnaroundS, echo_in = sent + 2 * kOneWayS + kTurnaroundS;
            const bool echo_lost = k % 11 == 5;
            const bool receive_lost = k % 7 == 3;
            feed_link(sync, snd_dev, snd_core, k, SyncRole::ReturnEgress, hw_stamp(rcv_dev, echo_out));
            if (!echo_lost) {
                feed_link(sync, snd_dev, snd_core, k, SyncRole::ReturnIngress, hw_stamp(snd_dev, echo_in));
            }
            if (!receive_lost && k != kLateRound) {
                receiver(k);
            }
            if (k == kLateArrival) {
                receiver(kLateRound);
            }
        }
    };
    burst(/*snd*/ 0, 0, /*rcv*/ 1, 0);
    burst(/*snd*/ 1, 1, /*rcv*/ 2, 0);
    sync.on_capture_end();

    ClockMap::Reader reader = map.reader();
    const auto placed_tsc = [&](int chip, double tau) {
        return map.place_host(reader, static_cast<uint32_t>(chip), wall_tick(chip, tau));
    };
    const auto placed_ns = [&](int chip, double tau) {
        return (static_cast<double>(placed_tsc(chip, tau) - kTsc0) - tsc(tau)) / kTicksPerNs;
    };
    const auto steady_ns = [&](int chip, double tau) {
        return static_cast<double>(*service().steady().ns(placed_tsc(chip, tau)) - kHost0);
    };
    for (double tau : {0.050, 0.150, 0.280}) {
        EXPECT_NEAR(placed_ns(0, tau), 0.0, kTolNs) << "chip0 root pre-switch, tau " << tau;
    }
    for (double tau : {0.400, 0.700, 0.950}) {
        EXPECT_NEAR(placed_ns(0, tau), 0.0, kTolNs) << "chip0 root post-switch, tau " << tau;
    }
    for (double tau : {0.050, 0.500, 0.950}) {
        EXPECT_NEAR(placed_ns(1, tau), 0.0, kTolNs) << "chip1 one hop, tau " << tau;
        EXPECT_NEAR(placed_ns(2, tau), 0.0, kTolNs) << "chip2 two hops, tau " << tau;
    }
    for (int chip : {0, 1, 2}) {
        for (double tau : {0.050, 0.500, 0.950}) {
            EXPECT_NEAR(steady_ns(chip, tau), host_ns(tau), 1.0) << "steady chip" << chip << ", tau " << tau;
        }
    }
}

TEST(StreamingProfilerSyncEngine, Retention) {
    constexpr uint32_t kNodes = 1u << 14;
    constexpr uint32_t kChip = 3;
    ClockMap map(kChip + 1, kNodes, {});
    ClockMap::Reader map_reader = map.reader();
    constexpr int64_t kStep = 1000;
    // The first segment is steeper than the rest, which alternate in slope, and no node's tangent matches a chord, so
    // the retired first node, a kept node and a mid-series chord each place differently. Every value is a multiple of
    // 2^-5 below 2^43, where a double resolves 2^-10, so the map's arithmetic on them is exact.
    const auto value = [](uint32_t node) {
        return node == 0 ? 7.5e12 - 62.5 : 7.5e12 + 31.25 * node + 3.90625 * (node % 2);
    };
    const auto tangent = [](uint32_t node) { return 0.041015625 + 0.001953125 * (node % 3); };
    const uint32_t node_count = kNodes + 1;
    for (uint32_t i = 0; i < node_count; i++) {
        map.append(kChip, SyncNode{.at = static_cast<int64_t>(i) * kStep, .value = value(i), .tangent = tangent(i)});
    }
    const auto root = [&](int64_t tick) { return map.place_root(map_reader, kChip, tick, 0.0); };
    const uint32_t mid = node_count / 2;
    const std::optional<double> newest = root(static_cast<int64_t>(node_count - 1) * kStep);
    const std::optional<double> mid_series = root(static_cast<int64_t>(mid) * kStep + kStep / 2);
    const std::optional<double> retired = root(0);
    ASSERT_TRUE(newest && mid_series && retired);
    EXPECT_NEAR(*newest, value(node_count - 1), 1e-3) << "the newest node";
    EXPECT_NEAR(*mid_series, (value(mid) + value(mid + 1)) / 2, 1e-3) << "a node mid-series";
    EXPECT_NEAR(*retired, value(1) - tangent(1) * kStep, 1e-3) << "the retired first node, on the oldest kept tangent";
}
