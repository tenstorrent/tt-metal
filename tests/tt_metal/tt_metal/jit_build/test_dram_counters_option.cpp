// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdlib>

#include "impl/profiler/dram_counters.hpp"
#include "llrt/rtoptions.hpp"

namespace tt::llrt {
namespace {

constexpr const char* kEnv = "TT_METAL_PROFILE_DRAM_COUNTERS";

class DramCountersOptionTest : public ::testing::Test {
protected:
    void SetUp() override { unsetenv(kEnv); }
    void TearDown() override { unsetenv(kEnv); }

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

}  // namespace
}  // namespace tt::tt_metal::dram_counters
