// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdlib>

#include "impl/profiler/eth_counters.hpp"
#include "llrt/rtoptions.hpp"

namespace tt::llrt {
namespace {

constexpr const char* kEthCountersEnv = "TT_METAL_PROFILE_ETH_COUNTERS";

class EthCountersOptionTest : public ::testing::Test {
protected:
    void SetUp() override { unsetenv(kEthCountersEnv); }
    void TearDown() override { unsetenv(kEthCountersEnv); }

    static RunTimeOptions with(const char* value) {
        setenv(kEthCountersEnv, value, 1);
        return RunTimeOptions{};
    }
};

TEST_F(EthCountersOptionTest, DisabledByDefault) {
    RunTimeOptions opts;
    EXPECT_FALSE(opts.get_profiler_eth_counters_enabled());
    EXPECT_FALSE(opts.get_profiler_enabled());
}

TEST_F(EthCountersOptionTest, OneEnablesWithCalibration) {
    RunTimeOptions opts = with("1");
    EXPECT_TRUE(opts.get_profiler_eth_counters_enabled());
    EXPECT_TRUE(opts.get_profiler_eth_counters_calibrate());
}

TEST_F(EthCountersOptionTest, RawSkipsCalibration) {
    RunTimeOptions opts = with("raw");
    EXPECT_TRUE(opts.get_profiler_eth_counters_enabled());
    EXPECT_FALSE(opts.get_profiler_eth_counters_calibrate());
}

TEST_F(EthCountersOptionTest, ZeroStaysDisabled) { EXPECT_FALSE(with("0").get_profiler_eth_counters_enabled()); }

TEST_F(EthCountersOptionTest, DoesNotTouchTheOtherCounterOptions) {
    RunTimeOptions opts = with("1");
    EXPECT_FALSE(opts.get_profiler_dram_counters_enabled());
    EXPECT_FALSE(opts.get_profiler_noc_counters_enabled());
    EXPECT_FALSE(opts.get_profiler_riscv_pmc_enabled());
    EXPECT_FALSE(opts.get_profiler_enabled());
}

}  // namespace
}  // namespace tt::llrt

namespace tt::tt_metal::eth_counters {
namespace {

namespace ec = llk::perf::eth;

size_t index_of(const Plan& plan, ec::EthCounter counter, Kind kind, int queue) {
    for (size_t i = 0; i < plan.items.size(); i++) {
        const Item& item = plan.items[i];
        if (item.counter == counter && item.kind == kind && item.queue == queue) {
            return i;
        }
    }
    ADD_FAILURE() << "item not in the plan";
    return 0;
}

TEST(EthCountersPlan, BlackholeReadsThreeQueuesEachWay) {
    const auto plan = make_plan(ec::Arch::BLACKHOLE, 14, true);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->items.size(), 3u * 5u + 3u * 5u + 25u);
    EXPECT_EQ(plan->reading_size(), 14u * plan->items.size());
    const Item& resend2 = plan->items[index_of(*plan, ec::EthCounter::TXQ_RESEND_CNT, Kind::TX_QUEUE, 2)];
    EXPECT_EQ(resend2.address, 0xFFB92068u);
    const Item& drop1 = plan->items[index_of(*plan, ec::EthCounter::RXQ_PKT_DROP_CNT, Kind::RX_QUEUE, 1)];
    EXPECT_EQ(drop1.address, 0xFFB9504Cu);
    EXPECT_TRUE(is_register(drop1));
}

TEST(EthCountersPlan, WormholeReadsTwoQueuesEachWay) {
    const auto plan = make_plan(ec::Arch::WORMHOLE, 16, true);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->items.size(), 2u * 5u + 2u * 4u + 4u);
    const Item& rx = plan->items[index_of(*plan, ec::EthCounter::RXQ_WORD_CNT, Kind::RX_QUEUE, 1)];
    EXPECT_EQ(rx.address, 0xFFB93014u);
}

TEST(EthCountersPlan, OneBlockReadCoversTheL1Items) {
    const auto bh = make_plan(ec::Arch::BLACKHOLE, 1, true);
    ASSERT_TRUE(bh.has_value());
    EXPECT_EQ(bh->l1_base, 0x7CE00u);
    EXPECT_EQ(bh->l1_bytes, 0xD0u);
    for (const Item& item : bh->items) {
        if (item.kind == Kind::L1) {
            EXPECT_GE(item.address, bh->l1_base);
            EXPECT_LE(item.address + 4u * item.words, bh->l1_base + bh->l1_bytes);
        }
    }
    const auto wh = make_plan(ec::Arch::WORMHOLE, 1, true);
    ASSERT_TRUE(wh.has_value());
    EXPECT_EQ(wh->l1_base, 0x1EDCu);
    EXPECT_EQ(wh->l1_bytes, 0x1FA0u - 0x1EDCu);
}

TEST(EthCountersPlan, OldFirmwareDropsTheL1Items) {
    const auto plan = make_plan(ec::Arch::BLACKHOLE, 14, false);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->items.size(), 30u);
    EXPECT_EQ(plan->l1_bytes, 0u);
    for (const Item& item : plan->items) {
        EXPECT_TRUE(is_register(item));
    }
}

TEST(EthCountersPlan, QuasarHasNoMap) { EXPECT_FALSE(make_plan(ec::Arch::QUASAR, 8, true).has_value()); }

TEST(EthCountersLayout, QueueAddresses) {
    EXPECT_EQ(ec::tx_queue_address(ec::Arch::BLACKHOLE, 1, 0x40), 0xFFB91040u);
    EXPECT_EQ(ec::rx_queue_address(ec::Arch::BLACKHOLE, 2, 0x4C), 0xFFB9604Cu);
    EXPECT_EQ(ec::rx_queue_address(ec::Arch::WORMHOLE, 0, 0x14), 0xFFB92014u);
    EXPECT_EQ(ec::layout_for(ec::Arch::BLACKHOLE).txq_word_bytes, 96u);
    EXPECT_EQ(ec::layout_for(ec::Arch::WORMHOLE).txq_word_bytes, 16u);
}

Reading flat(const Plan& plan, uint64_t value) {
    Reading r;
    r.values.assign(plan.reading_size(), value);
    return r;
}

TEST(EthCountersDiff, ThirtyTwoBitValuesWrapAndZeroIsSkipped) {
    const auto plan = make_plan(ec::Arch::BLACKHOLE, 2, false);
    ASSERT_TRUE(plan.has_value());
    Reading before = flat(*plan, 100), after = flat(*plan, 100);
    const size_t words = index_of(*plan, ec::EthCounter::TXQ_WORD_CNT, Kind::TX_QUEUE, 1);
    const size_t per_core = plan->items.size();
    before.values[per_core + words] = 0xFFFFFFFEu;  // core 1 wraps
    after.values[per_core + words] = 3;
    after.values[words] += 7;  // core 0
    uint32_t flags = 0;
    const auto deltas = diff(*plan, before, after, &flags);
    ASSERT_EQ(deltas.size(), 2u);
    EXPECT_EQ(deltas[0].core, 0u);
    EXPECT_EQ(deltas[0].item, words);
    EXPECT_EQ(deltas[0].count, 7u);
    EXPECT_EQ(deltas[1].core, 1u);
    EXPECT_EQ(deltas[1].count, 5u);
    EXPECT_EQ(flags, 0u);
}

TEST(EthCountersDiff, SixtyFourBitValuesDoNotWrapAndAResetIsFlagged) {
    const auto plan = make_plan(ec::Arch::BLACKHOLE, 1, true);
    ASSERT_TRUE(plan.has_value());
    const size_t frames = index_of(*plan, ec::EthCounter::FW_FRAMES_TXD, Kind::L1, NO_QUEUE);
    const size_t bytes = index_of(*plan, ec::EthCounter::FW_BYTES_TXD, Kind::L1, NO_QUEUE);
    Reading before = flat(*plan, 5), after = flat(*plan, 5);
    before.values[frames] = 0xFFFFFFFFull;
    after.values[frames] = 0x100000010ull;  // crosses the 32-bit boundary
    before.values[bytes] = 1000;
    after.values[bytes] = 10;  // went down
    after.values[index_of(*plan, ec::EthCounter::FW_LINK_POLL_ALIVE_COUNT, Kind::L1, NO_QUEUE)] += 1;
    uint32_t flags = 0;
    const auto deltas = diff(*plan, before, after, &flags);
    ASSERT_EQ(deltas.size(), 2u);  // the poll counter and the frames
    EXPECT_EQ(deltas[1].item, frames);
    EXPECT_EQ(deltas[1].count, 0x11u);
    EXPECT_EQ(flags, FLAG_RESET);
}

TEST(EthCountersDiff, LinkCountersAreDroppedWhenTheFirmwarePollCounterDidNotMove) {
    const auto plan = make_plan(ec::Arch::BLACKHOLE, 1, true);
    ASSERT_TRUE(plan.has_value());
    const size_t poll = index_of(*plan, ec::EthCounter::FW_LINK_POLL_ALIVE_COUNT, Kind::L1, NO_QUEUE);
    const size_t frames = index_of(*plan, ec::EthCounter::FW_FRAMES_TXD, Kind::L1, NO_QUEUE);
    const size_t words = index_of(*plan, ec::EthCounter::TXQ_WORD_CNT, Kind::TX_QUEUE, 0);
    Reading before = flat(*plan, 5), after = flat(*plan, 5);
    after.values[frames] += 9;
    after.values[words] += 4;
    uint32_t flags = 0;
    auto deltas = diff(*plan, before, after, &flags);
    ASSERT_EQ(deltas.size(), 1u);
    EXPECT_EQ(deltas[0].item, words);
    EXPECT_EQ(flags, FLAG_STALE);

    after.values[poll] += 1;
    flags = 0;
    deltas = diff(*plan, before, after, &flags);
    ASSERT_EQ(deltas.size(), 3u);
    EXPECT_EQ(flags, 0u);
}

TEST(EthCountersDiff, OnlyTheStaleCoreLosesItsLinkCounters) {
    const auto plan = make_plan(ec::Arch::BLACKHOLE, 2, true);
    ASSERT_TRUE(plan.has_value());
    const size_t per_core = plan->items.size();
    const size_t poll = index_of(*plan, ec::EthCounter::FW_LINK_POLL_ALIVE_COUNT, Kind::L1, NO_QUEUE);
    const size_t frames = index_of(*plan, ec::EthCounter::FW_FRAMES_TXD, Kind::L1, NO_QUEUE);
    Reading before = flat(*plan, 5), after = flat(*plan, 5);
    after.values[frames] += 9;           // core 0 is stale
    after.values[per_core + poll] += 1;  // core 1 refreshed
    after.values[per_core + frames] += 3;
    uint32_t flags = 0;
    const auto deltas = diff(*plan, before, after, &flags);
    ASSERT_EQ(deltas.size(), 2u);
    EXPECT_EQ(deltas[0].core, 1u);
    EXPECT_EQ(deltas[0].item, poll);
    EXPECT_EQ(deltas[1].core, 1u);
    EXPECT_EQ(deltas[1].item, frames);
    EXPECT_EQ(deltas[1].count, 3u);
    EXPECT_EQ(flags, FLAG_STALE);
}

TEST(EthCountersRun, RegisterTotalsKeepEveryCounterPeriod) {
    const auto plan = make_plan(ec::Arch::BLACKHOLE, 1, true);
    ASSERT_TRUE(plan.has_value());
    const size_t words = index_of(*plan, ec::EthCounter::TXQ_WORD_CNT, Kind::TX_QUEUE, 0);
    const size_t frames = index_of(*plan, ec::EthCounter::FW_FRAMES_TXD, Kind::L1, NO_QUEUE);
    std::vector<uint64_t> totals(plan->reading_size(), 0);
    Reading a = flat(*plan, 0), b = flat(*plan, 0), c = flat(*plan, 0);
    b.values[words] = 0xC0000000u;  // 3 Gi words, then 2 Gi more that wrap the register
    c.values[words] = 0x40000000u;
    c.values[frames] = 7;
    add_register_deltas(*plan, a, b, totals);
    add_register_deltas(*plan, b, c, totals);
    EXPECT_EQ(totals[words], 0x140000000ull);
    EXPECT_EQ(totals[frames], 0u);  // L1 values are 64-bit and come from the open and close readings
}

TEST(EthCountersNames, EveryCounterOfTheTablesHasAName) {
    for (const ec::Arch arch : {ec::Arch::BLACKHOLE, ec::Arch::WORMHOLE}) {
        const auto plan = make_plan(arch, 1, true);
        ASSERT_TRUE(plan.has_value());
        for (const Item& item : plan->items) {
            EXPECT_NE(counter_name(item.counter), "UNDEF");
        }
    }
}

}  // namespace
}  // namespace tt::tt_metal::eth_counters
