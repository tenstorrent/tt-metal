// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Deterministic tests of the host/device clock mapping the Tracy consumer places zones with. They need no device: the
// device clock is a known function of host time, whose rate falls with load the way a real one does, and the map is fed
// anchors taken from it.

#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "tt_metal/impl/realtime_profiler/realtime_profiler_clock_map.hpp"

namespace tt::tt_metal {
namespace {

constexpr int64_t kBaseHost = 5'000'000'000;       // host ticks (1 tick = 1 ns) at the first anchor
constexpr uint64_t kBaseWall = 1'000'000'000'000;  // device cycles at the first anchor
constexpr double kNominalCyclesPerTick = 1.35;     // what the device runs at before it is loaded
constexpr double kFinalCyclesPerTick = 1.18;       // and where load settles it
constexpr double kTauTicks = 3.0e9;                // how fast it gets there
constexpr int64_t kAnchorPeriod = 100'000'000;     // 100 ms

// Device cycles after the first anchor, at `t` host ticks after it: the integral of a rate falling from nominal to
// final.
double wall_after(double t) {
    const double swing = kNominalCyclesPerTick - kFinalCyclesPerTick;
    return kFinalCyclesPerTick * t + swing * kTauTicks * (1.0 - std::exp(-t / kTauTicks));
}

uint64_t wall_at(double t) { return kBaseWall + static_cast<uint64_t>(std::llround(wall_after(t))); }

// The host time at which the device clock reads `wall`, by bisection on the known function.
double host_after_for_wall(uint64_t wall) {
    double lo = 0.0, hi = 1.0e12;
    const double target = static_cast<double>(wall - kBaseWall);
    for (int i = 0; i < 100; ++i) {
        const double mid = 0.5 * (lo + hi);
        (wall_after(mid) < target ? lo : hi) = mid;
    }
    return 0.5 * (lo + hi);
}

DeviceClockMap make_map(int num_anchors) {
    DeviceClockMap map;
    map.start(kBaseHost, kBaseWall, 1.0 / kNominalCyclesPerTick);
    for (int k = 1; k <= num_anchors; ++k) {
        const double t = static_cast<double>(k) * static_cast<double>(kAnchorPeriod);
        map.add_anchor(kBaseHost + static_cast<int64_t>(t), wall_at(t));
    }
    return map;
}

TEST(RealtimeProfilerClockMap, ConstantRateMapsExactly) {
    DeviceClockMap map;
    constexpr double cycles_per_tick = 1.25;
    map.start(kBaseHost, kBaseWall, 1.0 / cycles_per_tick);
    for (int k = 1; k <= 6; ++k) {
        const int64_t t = k * kAnchorPeriod;
        map.add_anchor(kBaseHost + t, kBaseWall + static_cast<uint64_t>(std::llround(t * cycles_per_tick)));
    }
    for (double t = 0.05e9; t < 0.45e9; t += 7.3e6) {
        const uint64_t wall = kBaseWall + static_cast<uint64_t>(std::llround(t * cycles_per_tick));
        ASSERT_TRUE(map.covers(wall));
        EXPECT_NEAR(map.host_ticks(wall), t, 2.0) << "at " << t;
    }
}

TEST(RealtimeProfilerClockMap, FollowsARateThatChangesWithLoad) {
    const DeviceClockMap map = make_map(40);  // 4 s of 100 ms anchors across the steepest part of the ramp
    double worst = 0.0;
    for (double t = 0.2e9; t < 3.8e9; t += 3.1e6) {
        const uint64_t wall = wall_at(t);
        ASSERT_TRUE(map.covers(wall));
        worst = std::max(worst, std::abs(map.host_ticks(wall) - t));
    }
    // Within a microsecond of where the device clock really was.
    EXPECT_LT(worst, 1000.0) << "worst error in ns";
}

TEST(RealtimeProfilerClockMap, BackToBackZonesNeverOverlapOrCollapse) {
    const DeviceClockMap map = make_map(40);
    // Contiguous 144 us programs, as a saturated queue runs them: each starts where the last ended.
    uint64_t start = wall_at(0.3e9);
    double previous_end = map.host_ticks(start);
    int checked = 0;
    for (; start < wall_at(3.7e9); ++checked) {
        const uint64_t end = start + 190'000;  // about 144 us at 1.3 GHz
        ASSERT_TRUE(map.covers(end));
        const double mapped_start = map.host_ticks(start);
        const double mapped_end = map.host_ticks(end);
        ASSERT_GE(mapped_start, previous_end) << "zone " << checked << " starts before the last one ended";
        ASSERT_GT(mapped_end, mapped_start) << "zone " << checked << " collapsed";
        previous_end = mapped_end;
        start = end;
    }
    EXPECT_GT(checked, 15000);
}

TEST(RealtimeProfilerClockMap, AZoneStaysWhereItWasPlaced) {
    DeviceClockMap map = make_map(10);
    const uint64_t wall = wall_at(0.45e9);
    ASSERT_TRUE(map.covers(wall));
    const double placed = map.host_ticks(wall);
    // Later anchors must not move a zone that is already drawn: its segment's curve is fixed once both ends are final.
    for (int k = 11; k <= 40; ++k) {
        const double t = static_cast<double>(k) * static_cast<double>(kAnchorPeriod);
        map.add_anchor(kBaseHost + static_cast<int64_t>(t), wall_at(t));
    }
    EXPECT_NEAR(map.host_ticks(wall), placed, 1.0e-6 * placed);
}

TEST(RealtimeProfilerClockMap, TrimKeepsAnchorsADelayedZoneNeeds) {
    DeviceClockMap kept = make_map(40);
    DeviceClockMap reference = make_map(40);
    const uint64_t delayed = wall_at(0.55e9);  // a zone the consumer has not placed yet, 3+ anchors behind
    const double expected = reference.host_ticks(delayed);

    kept.trim(delayed);
    EXPECT_NEAR(kept.host_ticks(delayed), expected, 1.0e-6 * expected) << "a waiting zone must keep its curve";
    EXPECT_GT(kept.num_final_anchors(), 30u) << "anchors a waiting zone needs were dropped";

    // With nothing waiting only the newest segment is kept, and the oldest waiting zone is what makes the difference.
    DeviceClockMap trimmed = make_map(40);
    trimmed.trim(std::nullopt);
    EXPECT_EQ(trimmed.num_final_anchors(), 2u);
    EXPECT_GT(std::abs(trimmed.host_ticks(delayed) - expected), 1000.0)
        << "a zone older than every kept anchor is placed on a straight line, which the trim guard exists to avoid";
}

TEST(RealtimeProfilerClockMap, AnAnchorAfterTheLastZoneInterpolatesTheTail) {
    DeviceClockMap map = make_map(10);
    const uint64_t last_end = wall_at(1.0e9 + 30e6);  // a zone ending after the last anchor so far
    EXPECT_FALSE(map.covers(last_end)) << "no anchor after it yet, so it cannot be placed without extrapolating";

    // One more reading at the stop, after the last zone, and the zone is between two anchors.
    const double stop = 1.0e9 + 60e6;
    map.add_anchor(kBaseHost + static_cast<int64_t>(stop), wall_at(stop));
    map.finish();
    ASSERT_TRUE(map.covers(last_end));
    // The newest anchor has no later neighbour, so its slope is the last segment's own: a few microseconds of error,
    // against the hundreds a straight line past the last anchor would add as the rate moves.
    EXPECT_NEAR(map.host_ticks(last_end), 1.0e9 + 30e6, 10'000.0);
}

TEST(RealtimeProfilerClockMap, NeedsTwoFinalAnchorsBeforeItPlaces) {
    DeviceClockMap map;
    map.start(kBaseHost, kBaseWall, 1.0 / kNominalCyclesPerTick);
    EXPECT_FALSE(map.covers(kBaseWall + 1));
    map.add_anchor(kBaseHost + kAnchorPeriod, wall_at(1.0 * kAnchorPeriod));
    EXPECT_FALSE(map.covers(kBaseWall + 1));
    map.add_anchor(kBaseHost + 2 * kAnchorPeriod, wall_at(2.0 * kAnchorPeriod));
    EXPECT_TRUE(map.covers(wall_at(0.5 * kAnchorPeriod)));
}

TEST(RealtimeProfilerClockMap, IgnoresAnchorsThatDoNotMoveForward) {
    DeviceClockMap map = make_map(5);
    const double before = map.host_ticks(wall_at(0.25e9));
    // A bad reading: the device clock behind the last anchor, then the host clock behind it.
    map.add_anchor(kBaseHost + 6 * kAnchorPeriod, wall_at(2.0 * kAnchorPeriod));
    map.add_anchor(kBaseHost + 2 * kAnchorPeriod, wall_at(6.0 * kAnchorPeriod));
    EXPECT_DOUBLE_EQ(map.host_ticks(wall_at(0.25e9)), before);
    EXPECT_EQ(map.num_final_anchors(), 5u) << "a rejected reading must not become an anchor";
}

TEST(RealtimeProfilerClockMap, HostForWallHelperIsConsistent) {
    // Guards the test's own reference function.
    EXPECT_NEAR(host_after_for_wall(wall_at(1.5e9)), 1.5e9, 1.0);
}

}  // namespace
}  // namespace tt::tt_metal
