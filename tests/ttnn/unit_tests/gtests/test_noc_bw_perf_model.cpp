// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host-side lock for issue #58402. get_cycles_for_transaction_size turns the
// absolute GB/s NOC tables into cycles, and OpPerformanceModelGeneral turns
// those cycles back into nanoseconds at the live AICLK. The two clocks have to
// match: Wormhole 1000 MHz, Blackhole 1350 MHz. A 1.2 GHz Blackhole fallback
// reported ideal_ns 1.2/1.35 low.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::data_movement {
namespace {

// Same arithmetic as get_cycles_for_transaction_size, with the clock passed in
// so a test can show the retired 1.2 GHz Blackhole literal is no longer used.
uint32_t cycles_at(uint32_t num_transactions, uint32_t transaction_size, double bw_gbps, float clock_hz) {
    transaction_size = std::max(transaction_size, 16u);
    const float transaction_bw = static_cast<float>(bw_gbps);
    return static_cast<uint32_t>(std::ceil(
        static_cast<float>(num_transactions * transaction_size * clock_hz) / static_cast<float>(transaction_bw * 1e9)));
}

// OpPerformanceModelGeneral::ideal_compute_ns, with the live AICLK in MHz.
float ideal_ns_at(uint32_t cycles, int clock_mhz) {
    const float clock_rate_ghz = static_cast<float>(clock_mhz) / 1000.0f;
    return std::ceil(static_cast<float>(cycles) / clock_rate_ghz);
}

struct BwCase {
    bool is_dram;
    bool is_local;
    bool is_read;
    double wormhole_gbps;
    double blackhole_gbps;
};

// 2048 B rows of noc_dram_bw / noc_l1_read_bw / noc_l1_write_bw / noc_l1_local_bw.
constexpr BwCase kPaths[] = {
    {true, false, true, 28.411, 49.164},
    {false, false, true, 29.742, 47.977},
    {false, false, false, 27.811, 48.644},
    {false, true, true, 27.911, 46.455},
};

TEST(NocBwPerfModel, ClockMatchesLiveAiclkOnBothArches) {
    constexpr uint32_t kTransactionSize = 2048;
    constexpr uint32_t kNumTransactions = 1;
    for (const BwCase& path : kPaths) {
        const auto [wh_cycles, wh_latency] = get_cycles_for_transaction_size(
            kTransactionSize, path.is_dram, path.is_local, kNumTransactions, tt::ARCH::WORMHOLE_B0, path.is_read);
        const auto [bh_cycles, bh_latency] = get_cycles_for_transaction_size(
            kTransactionSize, path.is_dram, path.is_local, kNumTransactions, tt::ARCH::BLACKHOLE, path.is_read);

        EXPECT_EQ(wh_cycles, cycles_at(kNumTransactions, kTransactionSize, path.wormhole_gbps, 1.0e9f))
            << "wormhole path dram=" << path.is_dram << " local=" << path.is_local << " read=" << path.is_read;
        EXPECT_EQ(bh_cycles, cycles_at(kNumTransactions, kTransactionSize, path.blackhole_gbps, 1.35e9f))
            << "blackhole path dram=" << path.is_dram << " local=" << path.is_local << " read=" << path.is_read;
        // The bug: Blackhole cycles were computed at 1.2 GHz.
        EXPECT_NE(bh_cycles, cycles_at(kNumTransactions, kTransactionSize, path.blackhole_gbps, 1.2e9f))
            << "blackhole path dram=" << path.is_dram << " local=" << path.is_local << " read=" << path.is_read;

        if (path.is_dram) {
            EXPECT_EQ(wh_latency, 358u);
            EXPECT_EQ(bh_latency, 529u);
        } else if (path.is_local) {
            EXPECT_EQ(wh_latency, 56u);
            EXPECT_EQ(bh_latency, 88u);
        } else if (path.is_read) {
            EXPECT_EQ(wh_latency, 259u);
            EXPECT_EQ(bh_latency, 403u);
        } else {
            EXPECT_EQ(wh_latency, 256u);
            EXPECT_EQ(bh_latency, 404u);
        }
    }
}

TEST(NocBwPerfModel, BlackholeRoundTripNsMatchesTheGbpsTable) {
    // Large enough that one cycle of ceil is noise next to the old 11.1% bias.
    constexpr uint32_t kNumTransactions = 4096;
    constexpr uint32_t kTransactionSize = 2048;
    constexpr double kBlackholeDramGbps = 49.164;
    constexpr int kBlackholeAiclkMhz = 1350;

    const auto [cycles, latency] = get_cycles_for_transaction_size(
        kTransactionSize,
        /*is_dram=*/true,
        /*is_local=*/false,
        kNumTransactions,
        tt::ARCH::BLACKHOLE,
        /*is_read=*/true);
    EXPECT_EQ(latency, 529u);
    EXPECT_EQ(cycles, cycles_at(kNumTransactions, kTransactionSize, kBlackholeDramGbps, 1.35e9f));

    const double true_ns = static_cast<double>(kNumTransactions) * kTransactionSize / kBlackholeDramGbps;
    const double reported_ns = ideal_ns_at(cycles, kBlackholeAiclkMhz);
    const double retired_ns =
        ideal_ns_at(cycles_at(kNumTransactions, kTransactionSize, kBlackholeDramGbps, 1.2e9f), kBlackholeAiclkMhz);

    EXPECT_NEAR(reported_ns / true_ns, 1.0, 0.001);
    // 1.2/1.35 = 0.889. Guard the regression the other way too.
    EXPECT_LT(retired_ns / true_ns, 0.92);
}

TEST(NocBwPerfModel, WormholeRoundTripIsUnchangedAtOneGigahertz) {
    constexpr uint32_t kNumTransactions = 4096;
    constexpr uint32_t kTransactionSize = 2048;
    constexpr double kWormholeDramGbps = 28.411;

    const auto [cycles, latency] = get_cycles_for_transaction_size(
        kTransactionSize,
        /*is_dram=*/true,
        /*is_local=*/false,
        kNumTransactions,
        tt::ARCH::WORMHOLE_B0,
        /*is_read=*/true);
    EXPECT_EQ(latency, 358u);
    EXPECT_EQ(cycles, cycles_at(kNumTransactions, kTransactionSize, kWormholeDramGbps, 1.0e9f));

    const double true_ns = static_cast<double>(kNumTransactions) * kTransactionSize / kWormholeDramGbps;
    EXPECT_NEAR(ideal_ns_at(cycles, /*clock_mhz=*/1000) / true_ns, 1.0, 0.001);
}

TEST(NocBwPerfModel, SubBurstSizeClampsTo16AndOffTableSizeUsesBlackholeColumn) {
    const auto [clamped, clamped_latency] = get_cycles_for_transaction_size(
        /*transaction_size=*/1,
        /*is_dram=*/true,
        /*is_local=*/false,
        /*num_transactions=*/1,
        tt::ARCH::BLACKHOLE,
        /*is_read=*/true);
    const auto [at_16, at_16_latency] = get_cycles_for_transaction_size(
        /*transaction_size=*/16,
        /*is_dram=*/true,
        /*is_local=*/false,
        /*num_transactions=*/1,
        tt::ARCH::BLACKHOLE,
        /*is_read=*/true);
    EXPECT_EQ(clamped, at_16);
    EXPECT_EQ(clamped_latency, at_16_latency);
    // noc_dram_bw[16] Blackhole column.
    EXPECT_EQ(at_16, cycles_at(1, 16, 0.387, 1.35e9f));

    // 24 B is equidistant from 16 and 32; the model takes the upper entry.
    // noc_dram_bw[32] Blackhole column is 0.772, Wormhole is 0.868.
    const auto [off_table, off_table_latency] = get_cycles_for_transaction_size(
        /*transaction_size=*/24,
        /*is_dram=*/true,
        /*is_local=*/false,
        /*num_transactions=*/1,
        tt::ARCH::BLACKHOLE,
        /*is_read=*/true);
    EXPECT_EQ(off_table_latency, 529u);
    EXPECT_EQ(off_table, cycles_at(1, 24, 0.772, 1.35e9f));
    EXPECT_NE(off_table, cycles_at(1, 24, 0.868, 1.35e9f));
}

TEST(NocBwPerfModel, UnknownArchIsRejected) {
    for (const tt::ARCH arch : {tt::ARCH::QUASAR, tt::ARCH::Invalid}) {
        EXPECT_THROW(
            get_cycles_for_transaction_size(
                /*transaction_size=*/2048,
                /*is_dram=*/true,
                /*is_local=*/false,
                /*num_transactions=*/1,
                arch,
                /*is_read=*/true),
            std::runtime_error);
    }
}

}  // namespace
}  // namespace ttnn::operations::data_movement
