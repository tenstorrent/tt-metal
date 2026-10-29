// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <string>

#include <gtest/gtest.h>

#include "impl/dispatch/host_transfer_stats.hpp"

namespace tt::tt_metal {
namespace {

using namespace std::chrono_literals;
using Clock = HostTransferStats::Clock;

TEST(HostTransferStatsTest, CPU_BytesTransfersAndBusyTimeAddUp) {
    HostTransferStats stats(true);
    const auto t0 = Clock::now();
    stats.begin(HostTransfer::BufferWriteFastDispatch);
    stats.end(HostTransfer::BufferWriteFastDispatch, t0, t0 + 10ms, 1000);
    stats.begin(HostTransfer::BufferWriteFastDispatch);
    stats.end(HostTransfer::BufferWriteFastDispatch, t0 + 20ms, t0 + 25ms, 500);

    const auto totals = stats.totals(HostTransfer::BufferWriteFastDispatch);
    EXPECT_EQ(totals.bytes, 1500u);
    EXPECT_EQ(totals.transfers, 2u);
    EXPECT_EQ(totals.busy, 15ms) << "the idle gap between transfers is not busy time";
    EXPECT_EQ(stats.totals(HostTransfer::BufferWriteSlowDispatch).transfers, 0u) << "kinds are counted separately";
}

TEST(HostTransferStatsTest, CPU_OverlappingTransfersCountOnceEvenWhenClosedOutOfOrder) {
    HostTransferStats stats(true);
    const auto t0 = Clock::now();
    stats.begin(HostTransfer::BufferReadFastDispatch);
    stats.begin(HostTransfer::BufferReadFastDispatch);
    // The later transfer closes first; together they span t0 to t0 + 30ms.
    stats.end(HostTransfer::BufferReadFastDispatch, t0 + 10ms, t0 + 30ms, 1);
    EXPECT_EQ(stats.totals(HostTransfer::BufferReadFastDispatch).busy, 20ms) << "an open stretch counts so far";
    stats.end(HostTransfer::BufferReadFastDispatch, t0, t0 + 20ms, 1);

    EXPECT_EQ(stats.totals(HostTransfer::BufferReadFastDispatch).busy, 30ms);
}

TEST(HostTransferStatsTest, CPU_TransferWithoutBytesAddsNoTime) {
    HostTransferStats stats(true);
    const auto t0 = Clock::now();
    stats.begin(HostTransfer::BufferWriteFastDispatch);
    stats.end(HostTransfer::BufferWriteFastDispatch, t0, t0 + 10ms, 0);

    const auto totals = stats.totals(HostTransfer::BufferWriteFastDispatch);
    EXPECT_EQ(totals.transfers, 0u);
    EXPECT_EQ(totals.busy, Clock::duration::zero());
    EXPECT_TRUE(stats.summary_rows().empty());
}

TEST(HostTransferStatsTest, CPU_ScopedHostTransferRecordsOnlyWhileEnabled) {
    HostTransferStats stats(false);
    {
        ScopedHostTransfer transfer(&stats, HostTransfer::SocketPush, 64);
    }
    {
        ScopedHostTransfer transfer(nullptr, HostTransfer::SocketPush, 64);
    }
    EXPECT_EQ(stats.totals(HostTransfer::SocketPush).transfers, 0u);

    stats.set_enabled(true);
    {
        ScopedHostTransfer transfer(&stats, HostTransfer::SocketPush, 64);
    }
    EXPECT_EQ(stats.totals(HostTransfer::SocketPush).bytes, 64u);
    EXPECT_EQ(stats.totals(HostTransfer::SocketPush).transfers, 1u);
}

TEST(HostTransferStatsTest, CPU_SummaryShowsRateForTimedKindsAndStagedBytesForSocketPull) {
    HostTransferStats stats(true);
    const auto t0 = Clock::now();
    stats.begin(HostTransfer::BufferWriteSlowDispatch);
    stats.end(HostTransfer::BufferWriteSlowDispatch, t0, t0 + 2s, 3'000'000'000);
    stats.add_bytes(HostTransfer::SocketPull, 2048);

    const auto rows = stats.summary_rows();
    ASSERT_EQ(rows.size(), 2u);
    EXPECT_EQ(rows[0].first, "buffer writes, slow dispatch");
    EXPECT_EQ(rows[0].second, "3.00 GB in 2.000 s (1.50 GB/s), 1 transfers");
    EXPECT_EQ(rows[1].first, "H2D socket writes, device pull");
    EXPECT_EQ(rows[1].second, "2.05 KB staged, 1 transfers");
}

TEST(HostTransferStatsTest, CPU_ResetClearsTotals) {
    HostTransferStats stats(true);
    const auto t0 = Clock::now();
    stats.begin(HostTransfer::CoreWriteFastDispatch);
    stats.end(HostTransfer::CoreWriteFastDispatch, t0, t0 + 1ms, 4);
    stats.reset();

    const auto totals = stats.totals(HostTransfer::CoreWriteFastDispatch);
    EXPECT_EQ(totals.bytes, 0u);
    EXPECT_EQ(totals.transfers, 0u);
    EXPECT_EQ(totals.busy, Clock::duration::zero());
}

}  // namespace
}  // namespace tt::tt_metal
