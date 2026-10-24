// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::data_movement {
namespace {

// OpPerformanceModelGeneral divides cycles by the live AICLK.
float ideal_ns_at(uint32_t cycles, int clock_mhz) {
    const float clock_rate_ghz = static_cast<float>(clock_mhz) / 1000.0f;
    return std::ceil(static_cast<float>(cycles) / clock_rate_ghz);
}

struct BwRow {
    bool is_dram;
    bool is_local;
    bool is_read;
    double wormhole_gbps;
    double blackhole_gbps;
    uint32_t wormhole_latency;
    uint32_t blackhole_latency;
};

// 2048 B rows of noc_dram_bw, noc_l1_read_bw, noc_l1_write_bw, noc_l1_local_bw.
// Clocks are tt::umd::wormhole::AICLK_BUSY_VAL (1000 MHz) and tt::umd::blackhole::AICLK_BUSY_VAL (1350 MHz).
constexpr BwRow kRows[] = {
    {true, false, true, 28.411, 49.164, 358, 529},
    {false, false, true, 29.742, 47.977, 259, 403},
    {false, false, false, 27.811, 48.644, 256, 404},
    {false, true, true, 27.911, 46.455, 56, 88},
};

TEST(NocBwPerfModel, Row2048RoundTripMatchesBytesOverGbps) {
    constexpr uint32_t kSize = 2048;
    constexpr uint32_t kNum = 4096;
    constexpr double kBytes = static_cast<double>(kNum) * kSize;
    for (const BwRow& row : kRows) {
        const auto [wh_cycles, wh_latency] =
            get_cycles_for_transaction_size(kSize, row.is_dram, row.is_local, kNum, tt::ARCH::WORMHOLE_B0, row.is_read);
        const auto [bh_cycles, bh_latency] =
            get_cycles_for_transaction_size(kSize, row.is_dram, row.is_local, kNum, tt::ARCH::BLACKHOLE, row.is_read);
        EXPECT_NEAR(ideal_ns_at(wh_cycles, /*clock_mhz=*/1000) / (kBytes / row.wormhole_gbps), 1.0, 0.001);
        EXPECT_NEAR(ideal_ns_at(bh_cycles, /*clock_mhz=*/1350) / (kBytes / row.blackhole_gbps), 1.0, 0.001);
        EXPECT_EQ(wh_latency, row.wormhole_latency);
        EXPECT_EQ(bh_latency, row.blackhole_latency);
    }
}

// noc_dram_bw Blackhole: 16 B is 0.387 GB/s. 24 B takes the 32 B entry, 0.772, not Wormhole's 0.868.
TEST(NocBwPerfModel, BelowMinSizeClampsAndTieUsesUpperBlackholeEntry) {
    constexpr uint32_t kNum = 4096;
    const auto [clamped, clamped_latency] = get_cycles_for_transaction_size(
        /*transaction_size=*/1, true, false, kNum, tt::ARCH::BLACKHOLE, true);
    const auto [at_16, at_16_latency] =
        get_cycles_for_transaction_size(/*transaction_size=*/16, true, false, kNum, tt::ARCH::BLACKHOLE, true);
    EXPECT_EQ(clamped, at_16);
    EXPECT_EQ(clamped_latency, at_16_latency);
    EXPECT_NEAR(ideal_ns_at(at_16, /*clock_mhz=*/1350) / (static_cast<double>(kNum) * 16.0 / 0.387), 1.0, 0.001);

    const auto [off_table, off_table_latency] =
        get_cycles_for_transaction_size(/*transaction_size=*/24, true, false, kNum, tt::ARCH::BLACKHOLE, true);
    EXPECT_EQ(off_table_latency, 529u);
    const double bytes = static_cast<double>(kNum) * 24.0;
    EXPECT_NEAR(ideal_ns_at(off_table, /*clock_mhz=*/1350) / (bytes / 0.772), 1.0, 0.001);
    EXPECT_GT(std::abs(ideal_ns_at(off_table, /*clock_mhz=*/1350) / (bytes / 0.868) - 1.0), 0.05);
}

TEST(NocBwPerfModel, UnknownArchIsRejected) {
    for (const tt::ARCH arch : {tt::ARCH::QUASAR, tt::ARCH::Invalid}) {
        EXPECT_THROW(
            get_cycles_for_transaction_size(
                /*transaction_size=*/2048, true, false, /*num_transactions=*/1, arch, true),
            std::runtime_error);
    }
}

}  // namespace
}  // namespace ttnn::operations::data_movement
