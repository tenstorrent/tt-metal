// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdlib>
#include <optional>
#include <set>
#include <string>

#include "impl/profiler/dram_counters.hpp"
#include "llrt/rtoptions.hpp"

namespace tt::llrt {
namespace {

constexpr const char* kEnv = "TT_METAL_PROFILE_DRAM_COUNTERS";
constexpr const char* kEnvController = "TT_METAL_PROFILE_DRAM_CONTROLLER";

class DramCountersOptionTest : public ::testing::Test {
protected:
    void SetUp() override {
        unsetenv(kEnv);
        unsetenv(kEnvController);
    }
    void TearDown() override {
        unsetenv(kEnv);
        unsetenv(kEnvController);
    }

    static RunTimeOptions with(const char* value) {
        setenv(kEnv, value, 1);
        return RunTimeOptions{};
    }
};

TEST_F(DramCountersOptionTest, DisabledByDefault) {
    RunTimeOptions opts;
    EXPECT_FALSE(opts.get_profiler_dram_counters_enabled());
    EXPECT_FALSE(opts.get_profiler_enabled());
}

TEST_F(DramCountersOptionTest, OneEnablesWithCalibration) {
    RunTimeOptions opts = with("1");
    EXPECT_TRUE(opts.get_profiler_dram_counters_enabled());
    EXPECT_TRUE(opts.get_profiler_dram_counters_calibrate());
}

TEST_F(DramCountersOptionTest, RawSkipsCalibration) {
    RunTimeOptions opts = with("raw");
    EXPECT_TRUE(opts.get_profiler_dram_counters_enabled());
    EXPECT_FALSE(opts.get_profiler_dram_counters_calibrate());
}

TEST_F(DramCountersOptionTest, ZeroStaysDisabled) { EXPECT_FALSE(with("0").get_profiler_dram_counters_enabled()); }

TEST_F(DramCountersOptionTest, DoesNotForceTheDeviceProfiler) { EXPECT_FALSE(with("1").get_profiler_enabled()); }

TEST_F(DramCountersOptionTest, ControllerOffByDefault) {
    EXPECT_FALSE(with("1").get_profiler_dram_controller_enabled());
}

TEST_F(DramCountersOptionTest, ControllerRidesOnTheTileCapture) {
    setenv(kEnvController, "1", 1);
    RunTimeOptions opts = with("raw");
    EXPECT_TRUE(opts.get_profiler_dram_controller_enabled());
    EXPECT_TRUE(opts.get_profiler_dram_counters_enabled());
}

TEST_F(DramCountersOptionTest, ControllerAloneIsRejected) {
    setenv(kEnvController, "1", 1);
    EXPECT_ANY_THROW(RunTimeOptions{});
}

}  // namespace
}  // namespace tt::llrt

namespace tt::tt_metal::dram_counters {
namespace {

namespace dc = llk::perf::dram;

TEST(DramCountersPlan, BlackholeCapturesThreeCountersPerNoc) {
    const auto plan = make_plan(dc::Arch::BLACKHOLE, 21);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->num_nocs, 2u);
    ASSERT_EQ(plan->counters.size(), 3u);
    EXPECT_EQ(plan->counters[0], dc::DramCounter::NIU_SLV_RD_DATA_WORD_SENT);
    EXPECT_EQ(plan->offsets[0], 0x2CCu);
    EXPECT_EQ(plan->reading_size(), 21u * 2u * 3u);
    EXPECT_TRUE(is_read_counter(plan->counters[0]));
    EXPECT_TRUE(is_write_counter(plan->counters[1]));
    EXPECT_TRUE(is_write_counter(plan->counters[2]));
}

TEST(DramCountersPlan, QuasarHasNoMap) { EXPECT_FALSE(make_plan(dc::Arch::QUASAR, 8).has_value()); }

TEST(DramCountersLayout, BlackholeAddressIsTheWorkerOne) {
    const auto niu = dc::niu_table(dc::Arch::BLACKHOLE);
    ASSERT_EQ(niu.size, 6u);
    EXPECT_EQ(niu.data[1].first, dc::DramCounter::NIU_SLV_RD_DATA_WORD_SENT);
    for (uint32_t y : {0u, 1u, 2u, 11u}) {
        EXPECT_EQ(dc::niu_address(dc::Arch::BLACKHOLE, 0, y, niu.data[1].second), 0xFFB202CCull);
        EXPECT_EQ(dc::niu_address(dc::Arch::BLACKHOLE, 1, y, niu.data[1].second), 0xFFB302CCull);
    }
    EXPECT_EQ(dc::layout_for(dc::Arch::BLACKHOLE).word_bytes, 64u);
}

TEST(DramCountersLayout, WormholeDramTileInstances) {
    const auto niu = dc::niu_table(dc::Arch::WORMHOLE);
    ASSERT_EQ(niu.size, 6u);
    const uint32_t rd = niu.data[1].second;
    EXPECT_EQ(rd, 0x2CCu);
    EXPECT_EQ(dc::niu_address(dc::Arch::WORMHOLE, 0, 1, rd), 0x1000802CCull);
    EXPECT_EQ(dc::niu_address(dc::Arch::WORMHOLE, 0, 2, rd), 0x1000902CCull);
    EXPECT_EQ(dc::niu_address(dc::Arch::WORMHOLE, 0, 0, rd), 0x1000A02CCull);
    EXPECT_EQ(dc::niu_address(dc::Arch::WORMHOLE, 0, 3, rd), 0x1000A02CCull);
    EXPECT_EQ(dc::niu_address(dc::Arch::WORMHOLE, 1, 1, rd), 0x1000882CCull);
    EXPECT_EQ(dc::layout_for(dc::Arch::WORMHOLE).word_bytes, 32u);
}

namespace {
// Two tiles, readings with the per-tile read index of every value (3 counters x 2 NoCs per tile, reads in order).
Reading reading_with_indices(const Plan& plan, uint32_t first_index) {
    Reading r;
    r.values.assign(plan.reading_size(), 100);
    for (size_t t = 0; t < plan.num_tiles; t++) {
        for (size_t k = 0; k < plan.num_nocs * plan.counters.size(); k++) {
            r.tile_reads.push_back(first_index + static_cast<uint32_t>(k));
        }
    }
    return r;
}
}  // namespace

TEST(DramCountersDiff, WrapsAndSkipsZero) {
    const auto plan = make_plan(dc::Arch::BLACKHOLE, 2);
    ASSERT_TRUE(plan.has_value());
    const Coupling none(2, 0);
    Reading before = reading_with_indices(*plan, 1), after = reading_with_indices(*plan, 7);
    after.values[1] += 7;                // tile 0, noc 0, nonposted write data
    before.values[3 + 2] = 0xFFFFFFFEu;  // tile 0, noc 1, posted write data wraps
    after.values[3 + 2] = 3;
    const auto deltas = diff(*plan, none, before, after);
    ASSERT_EQ(deltas.size(), 2u);
    EXPECT_EQ(deltas[0].noc, 0);
    EXPECT_EQ(deltas[0].counter, dc::DramCounter::NIU_SLV_NONPOSTED_WR_DATA_WORD_RECEIVED);
    EXPECT_EQ(deltas[0].words, 7u);
    EXPECT_EQ(deltas[1].noc, 1);
    EXPECT_EQ(deltas[1].counter, dc::DramCounter::NIU_SLV_POSTED_WR_DATA_WORD_RECEIVED);
    EXPECT_EQ(deltas[1].words, 5u);
}

TEST(DramCountersDiff, OwnReadsAreRemovedFromTheCoupledNocOnly) {
    const auto plan = make_plan(dc::Arch::BLACKHOLE, 1);
    ASSERT_TRUE(plan.has_value());
    const Coupling noc0_only = {1, 0};
    Reading before = reading_with_indices(*plan, 1), after = reading_with_indices(*plan, 1);
    // tile 0 read index of the read counters: noc 0 value 0 and noc 1 value 3.
    before.tile_reads[0] = 10;
    after.tile_reads[0] = 262;  // 252 own reads in between
    before.tile_reads[3] = 12;
    after.tile_reads[3] = 264;
    after.values[0] += 252 + 40;  // own reads plus 40 words of real traffic
    after.values[3] += 252;       // not coupled: all of it is real traffic
    const auto deltas = diff(*plan, noc0_only, before, after);
    ASSERT_EQ(deltas.size(), 2u);
    EXPECT_EQ(deltas[0].words, 40u);
    EXPECT_EQ(deltas[0].self_words, 252u);
    EXPECT_EQ(deltas[1].words, 252u);
    EXPECT_EQ(deltas[1].self_words, 0u);
}

TEST(DramCountersDiff, WriteCountersNeverGetSelfCorrection) {
    const auto plan = make_plan(dc::Arch::BLACKHOLE, 1);
    ASSERT_TRUE(plan.has_value());
    const Coupling both = {1, 1};
    Reading before = reading_with_indices(*plan, 1), after = reading_with_indices(*plan, 500);
    after.values[1] += 9;
    const auto deltas = diff(*plan, both, before, after);
    ASSERT_EQ(deltas.size(), 1u);
    EXPECT_EQ(deltas[0].words, 9u);
    EXPECT_EQ(deltas[0].self_words, 0u);
}

TEST(DramCountersController, OnlyBlackholeHasAMap) {
    const auto bh = dc::controller_layout_for(dc::Arch::BLACKHOLE);
    EXPECT_EQ(bh.register_window, 1ull << 40);
    EXPECT_GT(dc::controller_table(dc::Arch::BLACKHOLE).size, 0u);
    EXPECT_EQ(dc::controller_layout_for(dc::Arch::WORMHOLE).register_window, 0u);
    EXPECT_EQ(dc::controller_table(dc::Arch::WORMHOLE).size, 0u);
    EXPECT_EQ(dc::controller_table(dc::Arch::QUASAR).size, 0u);
}

TEST(DramCountersController, CountersAreDistinctMonitorRegisters) {
    const auto table = dc::controller_table(dc::Arch::BLACKHOLE);
    const auto l = dc::controller_layout_for(dc::Arch::BLACKHOLE);
    std::set<uint32_t> seen;
    std::set<std::string> names;
    for (size_t i = 0; i < table.size; i++) {
        EXPECT_TRUE(seen.insert(table.data[i].address).second) << table.data[i].name;
        EXPECT_TRUE(names.insert(table.data[i].name).second) << table.data[i].name;
        EXPECT_EQ(table.data[i].address & 3u, 0u);
        EXPECT_NE(table.data[i].address, l.interval);
        EXPECT_NE(table.data[i].address, l.interval_done);
    }
}

TEST(DramCountersController, FirstPortWithEveryAxiResetReleasedAnswers) {
    const auto l = dc::controller_layout_for(dc::Arch::BLACKHOLE);
    EXPECT_EQ(answering_port({0u, 0x10007u, 0x10007u}, l), std::optional<size_t>(1));
    EXPECT_EQ(answering_port({0x7u}, l), std::optional<size_t>(0));
    EXPECT_FALSE(answering_port({0u, 0x3u, 0x10000u}, l).has_value());
    EXPECT_FALSE(answering_port({}, l).has_value());
    EXPECT_FALSE(answering_port({0x7u}, dc::controller_layout_for(dc::Arch::WORMHOLE)).has_value());
}

TEST(DramCountersIdleFit, ConstantRateRecoversBothTerms) {
    // 5 words per snapshot pair plus 1000 words per second, sampled over 1 ms and 21 ms windows.
    const IdleFit fit = fit_idle(5.0 + 1.0, 0.001, 5.0 + 21.0, 0.021);
    EXPECT_NEAR(fit.per_pair, 5.0, 1e-9);
    EXPECT_NEAR(fit.per_second, 1000.0, 1e-6);
}

TEST(DramCountersIdleFit, NoTermGoesNegative) {
    const IdleFit fit = fit_idle(10.0, 0.001, 4.0, 0.021);
    EXPECT_EQ(fit.per_second, 0.0);
    EXPECT_EQ(fit.per_pair, 10.0);
    EXPECT_EQ(fit_idle(1.0, 0.001, 1001.0, 0.002).per_pair, 0.0);
}

}  // namespace
}  // namespace tt::tt_metal::dram_counters
