// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

#include "jit_build/build_cache_telemetry.hpp"
#include "jit_build/jit_build_cache.hpp"

namespace tt::tt_metal {

TEST(JitBuildCacheTests, BuildOnceRunsFnOnce) {
    auto& cache = JitBuildCache::inst();
    cache.clear();

    int runs = 0;
    EXPECT_TRUE(cache.build_once(0xabc, [&] { ++runs; }));
    EXPECT_EQ(runs, 1);

    EXPECT_FALSE(cache.build_once(0xabc, [&] { ++runs; }));
    EXPECT_EQ(runs, 1);
}

TEST(JitBuildCacheTests, DistinctHashesRunSeparately) {
    auto& cache = JitBuildCache::inst();
    cache.clear();

    int runs = 0;
    EXPECT_TRUE(cache.build_once(1, [&] { ++runs; }));
    EXPECT_TRUE(cache.build_once(2, [&] { ++runs; }));
    EXPECT_EQ(runs, 2);
}

class BuildCacheTelemetryTest : public ::testing::Test {
protected:
    void TearDown() override {
        auto& tel = BuildCacheTelemetry::inst();
        if (!tel.is_enabled()) {
            tel.enable();
        }
    }
};

TEST_F(BuildCacheTelemetryTest, DoubleEnablePreservesCounters) {
    auto& tel = BuildCacheTelemetry::inst();
    tel.disable();
    tel.enable();
    ASSERT_TRUE(tel.is_enabled());

    tel.record_jit_once_dedup();
    EXPECT_EQ(tel.get_jit_once_dedup_count(), 1u);

    tel.enable();
    tel.record_jit_once_dedup();
    EXPECT_EQ(tel.get_jit_once_dedup_count(), 2u);
}

// BuildCacheTelemetry is a process-wide singleton shared by every test in this binary, and tokens
// are never destroyed, so each test below uses metric names of its own and compares snapshots
// taken before and after the operation under test rather than absolute values.

TEST_F(BuildCacheTelemetryTest, RepeatRegistrationReturnsTheSameToken) {
    auto& tel = BuildCacheTelemetry::inst();
    auto& first = tel.get_or_register_metric("test.repeat_registration");
    auto& second = tel.get_or_register_metric("test.repeat_registration");

    EXPECT_EQ(&first, &second);
    EXPECT_EQ(first.name(), "test.repeat_registration");
}

TEST_F(BuildCacheTelemetryTest, SameNameFromTwoCallSitesAggregatesIntoOneStream) {
    auto& tel = BuildCacheTelemetry::inst();
    // The point of deduplication: two call sites that name the same metric must feed one set of
    // aggregates, rather than the later one shadowing the earlier in the dump.
    tel.get_or_register_metric("test.shared_stream").record(10.0);
    tel.get_or_register_metric("test.shared_stream").record(30.0);

    const TelemetryTokenData snap = tel.get_or_register_metric("test.shared_stream").snapshot();
    EXPECT_EQ(snap.count, 2u);
    EXPECT_DOUBLE_EQ(snap.total, 40.0);
    EXPECT_DOUBLE_EQ(snap.min_val, 10.0);
    EXPECT_DOUBLE_EQ(snap.max_val, 30.0);
}

TEST_F(BuildCacheTelemetryTest, UnitDefaultsToMsAndIsRemembered) {
    auto& tel = BuildCacheTelemetry::inst();
    EXPECT_EQ(tel.get_or_register_metric("test.default_unit").unit(), "ms");
    EXPECT_EQ(tel.get_or_register_metric("test.byte_unit", "B").unit(), "B");
    // The unit comes from the first registration; a matching repeat is accepted and changes nothing.
    EXPECT_EQ(tel.get_or_register_metric("test.byte_unit", "B").unit(), "B");
}

TEST_F(BuildCacheTelemetryTest, OmittingTheUnitIsALookupNotAnMsRegistration) {
    auto& tel = BuildCacheTelemetry::inst();
    auto& registered = tel.get_or_register_metric("test.unit_omitted_lookup", "B");

    // A caller that only wants the token should not have to restate the unit, and must not be
    // treated as registering the default "ms" over a byte metric.
    auto& looked_up = tel.get_or_register_metric("test.unit_omitted_lookup");
    EXPECT_EQ(&looked_up, &registered);
    EXPECT_EQ(looked_up.unit(), "B");
}

TEST_F(BuildCacheTelemetryTest, ConflictingUnitKeepsTheRegisteredOneWithoutAborting) {
    auto& tel = BuildCacheTelemetry::inst();
    auto& token = tel.get_or_register_metric("test.unit_conflict", "ms");
    token.record(7.0);

    // Two call sites naming one metric with different units is a bug: their values land in one
    // set of aggregates labelled with whichever unit was registered first. Telemetry is a
    // diagnostic subsystem though, so this warns and carries on rather than killing the process
    // -- a wrong unit in a log line should not fail a build.
    EXPECT_EQ(&tel.get_or_register_metric("test.unit_conflict", "B"), &token);
    EXPECT_EQ(tel.get_or_register_metric("test.unit_conflict", "B").unit(), "ms");

    // ...and the existing stream is left alone: no reset, and no sample added by the call that
    // named the wrong unit.
    const TelemetryTokenData snap = tel.get_or_register_metric("test.unit_conflict").snapshot();
    EXPECT_EQ(snap.count, 1u);
    EXPECT_DOUBLE_EQ(snap.total, 7.0);
}

TEST_F(BuildCacheTelemetryTest, PerTargetTokenBuildsMetricDotTargetKey) {
    // Both new call sites (fw/kernel link time and ELF size in build.cpp, the program_config_size
    // streams in program.cpp) depend on this key format.
    auto& a = per_target_telemetry_token("test.per_target", "targetA", "B");
    auto& b = per_target_telemetry_token("test.per_target", "targetB", "B");

    EXPECT_EQ(a.name(), "test.per_target.targetA");
    EXPECT_EQ(b.name(), "test.per_target.targetB");
    EXPECT_NE(&a, &b) << "two targets of one metric are separate streams";
    EXPECT_EQ(a.unit(), "B");

    EXPECT_EQ(&per_target_telemetry_token("test.per_target", "targetA", "B"), &a)
        << "same metric and target must resolve to the same token";
    EXPECT_EQ(&BuildCacheTelemetry::inst().get_or_register_metric("test.per_target.targetA"), &a)
        << "the per-target token is the same stream as its flattened name";
}

TEST_F(BuildCacheTelemetryTest, ConcurrentRegistrationOfOneNameYieldsOneToken) {
    auto& tel = BuildCacheTelemetry::inst();
    constexpr int num_threads = 16;
    constexpr int records_per_thread = 100;

    std::vector<std::thread> threads;
    std::vector<TelemetryToken*> observed(num_threads, nullptr);
    threads.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i] {
            auto& token = tel.get_or_register_metric("test.concurrent_registration");
            observed[i] = &token;
            for (int j = 0; j < records_per_thread; ++j) {
                token.record(1.0);
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    for (auto* token : observed) {
        EXPECT_EQ(token, observed[0]) << "a raced registration must not produce a second token";
    }
    EXPECT_EQ(observed[0]->snapshot().count, static_cast<uint32_t>(num_threads * records_per_thread));
}

TEST_F(BuildCacheTelemetryTest, TokenSurvivesDisableEnableAndStopsRecordingWhileDisabled) {
    auto& tel = BuildCacheTelemetry::inst();
    auto* token = &tel.get_or_register_metric("test.disable_cycle");
    token->record(1.0);

    tel.disable();
    // Same object across the cycle: callers cache the reference in a function-local static, so a
    // token replaced by disable()/enable() would leave them writing into a dead stream.
    EXPECT_EQ(&tel.get_or_register_metric("test.disable_cycle"), token);
    token->record(99.0);
    EXPECT_EQ(token->snapshot().count, 1u) << "record() must be a no-op while telemetry is disabled";

    tel.enable();
    EXPECT_EQ(&tel.get_or_register_metric("test.disable_cycle"), token);
    token->record(3.0);

    const TelemetryTokenData snap = token->snapshot();
    EXPECT_EQ(snap.count, 2u);
    EXPECT_DOUBLE_EQ(snap.total, 4.0);
}

TEST_F(BuildCacheTelemetryTest, MetricRegisteredWhileDisabledRecordsOnceEnabled) {
    auto& tel = BuildCacheTelemetry::inst();
    tel.disable();
    auto& token = tel.get_or_register_metric("test.registered_while_disabled");
    token.record(5.0);
    EXPECT_EQ(token.snapshot().count, 0u);

    tel.enable();
    token.record(5.0);
    EXPECT_EQ(token.snapshot().count, 1u);
}

// --- JIT build window ---

class JitBuildWindowTest : public BuildCacheTelemetryTest {
protected:
    void SetUp() override {
        auto& tel = BuildCacheTelemetry::inst();
        // disable()/enable() rebuilds impl_, which is where the window endpoints live, so this
        // clears any span noted by an earlier test.
        tel.disable();
        tel.enable();
    }

    static TelemetryTokenData window_snapshot() {
        return BuildCacheTelemetry::inst().get_or_register_metric("jit_build_window").snapshot();
    }

    // dump_metrics() is what collapses the window endpoints into the single sample the metric
    // carries. Token aggregates outlive disable()/enable(), so return the span of the sample this
    // call added rather than reading the token's absolute totals.
    static double dump_and_get_window_ms() {
        const TelemetryTokenData before = window_snapshot();
        BuildCacheTelemetry::inst().dump_metrics();
        const TelemetryTokenData after = window_snapshot();
        EXPECT_EQ(after.count, before.count + 1u) << "dump_metrics() should record exactly one window sample";
        return after.total - before.total;
    }
};

TEST_F(JitBuildWindowTest, WindowMeasuresBuildSpanNotProcessLifetime) {
    using namespace std::chrono_literals;
    auto& tel = BuildCacheTelemetry::inst();

    // A build that ran for 50 ms, ten seconds before the dump. This is the regression that a
    // static-lifetime ScopedTelemetryTimer had: it stopped at process exit, so everything the
    // process did after the last compile was charged to the build.
    const auto build_start = std::chrono::steady_clock::now() - 10s;
    tel.note_build_window(build_start, build_start + 50ms);

    EXPECT_NEAR(dump_and_get_window_ms(), 50.0, 1.0);
}

TEST_F(JitBuildWindowTest, WindowSpansEarliestEntryToLatestExit) {
    using namespace std::chrono_literals;
    auto& tel = BuildCacheTelemetry::inst();

    // Overlapping builds on the thread pool, noted out of order as they finish. The window is
    // the union's extent -- summing the individual durations would over-count the overlap.
    const auto t0 = std::chrono::steady_clock::now();
    tel.note_build_window(t0 + 20ms, t0 + 60ms);
    tel.note_build_window(t0, t0 + 40ms);
    tel.note_build_window(t0 + 10ms, t0 + 30ms);

    EXPECT_NEAR(dump_and_get_window_ms(), 60.0, 1.0);
}

TEST_F(JitBuildWindowTest, ConcurrentBuildsWidenTheWindowToTheirExtent) {
    using namespace std::chrono_literals;
    auto& tel = BuildCacheTelemetry::inst();
    constexpr int num_threads = 16;

    // Both endpoints are read-modify-write, so hammer them from every thread at once: thread i
    // covers [i ms, i+5 ms], making the union [0 ms, 20 ms].
    const auto base = std::chrono::steady_clock::now();
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i] {
            for (int j = 0; j < 200; ++j) {
                tel.note_build_window(base + std::chrono::milliseconds(i), base + std::chrono::milliseconds(i + 5));
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_NEAR(dump_and_get_window_ms(), 20.0, 1.0);
}

TEST_F(JitBuildWindowTest, NoBuildActivityRecordsNoWindow) {
    const uint32_t count_before = window_snapshot().count;

    BuildCacheTelemetry::inst().dump_metrics();

    EXPECT_EQ(window_snapshot().count, count_before) << "a process that never built should report no window";
}

TEST_F(JitBuildWindowTest, ScopedBuildWindowContributesOnScopeExit) {
    const uint32_t count_before = window_snapshot().count;
    {
        ScopedBuildWindow window;
        BuildCacheTelemetry::inst().dump_metrics();
        EXPECT_EQ(window_snapshot().count, count_before) << "an open window has no end yet, so nothing to report";
    }
    EXPECT_GE(dump_and_get_window_ms(), 0.0);
}

TEST_F(JitBuildWindowTest, ScopedBuildWindowContributesWhenScopeThrows) {
    // JitBuildState::build() throws on compile failure, and a failed build still occupied wall
    // clock time that belongs inside the window.
    EXPECT_THROW(
        {
            ScopedBuildWindow window;
            throw std::runtime_error("build failed");
        },
        std::runtime_error);

    EXPECT_GE(dump_and_get_window_ms(), 0.0);
}

TEST_F(JitBuildWindowTest, WindowIsNotRecordedWhileTelemetryIsDisabled) {
    using namespace std::chrono_literals;
    auto& tel = BuildCacheTelemetry::inst();
    const uint32_t count_before = window_snapshot().count;
    tel.disable();

    const auto t0 = std::chrono::steady_clock::now();
    tel.note_build_window(t0, t0 + 50ms);
    tel.dump_metrics();

    tel.enable();
    EXPECT_EQ(window_snapshot().count, count_before);
}

}  // namespace tt::tt_metal
