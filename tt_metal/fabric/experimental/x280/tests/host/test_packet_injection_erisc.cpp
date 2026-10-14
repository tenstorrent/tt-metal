// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// tests/host/test_packet_injection_erisc.cpp -- pure host tests for the ERISC
// packet-header ingress member: its worker family and its writer selection.

#include <gtest/gtest.h>

#include <vector>

#include "tests/device/suites/packet_injection/regions.hpp"

using namespace x280::harness::regions::packet_injection;

namespace {

Config EthernetConfig(uint32_t writers) {
    Config cfg{};
    cfg.mode = PI_MODE_THROUGHPUT;
    cfg.family = WorkerFamily::Ethernet;
    cfg.tile_count = writers;
    for (uint32_t y = 0; y < writers; ++y) {
        cfg.ordered_tiles.push_back(LogicalCoord{0u, y});
    }
    return cfg;
}

}  // namespace

// The family enum's values are not multipliers: Ethernet is 3, and a worker
// count derived from it would admit three producers per tile.
TEST(PacketInjectionErisc, EthernetFamilyIsOneProducerPerTile) {
    const Config cfg = EthernetConfig(kMaxEthernetWriters);
    EXPECT_EQ(worker_count(cfg), kMaxEthernetWriters);

    Capacity cap{};
    cap.lim_backed_bytes = kRequiredLimBackedBytes;
    cap.l1_span_limit = 455264u;
    cap.l1_window_base = 87872u;
    cap.producer_timer_hz = 1350000000ULL;
    cap.mtime_hz = x280::harness::measurement::kX280MtimeHz;
    const Admission adm = validate_config(cfg, cap, cap.l1_window_base);
    ASSERT_TRUE(adm.ok()) << adm.describe();
    EXPECT_EQ(adm.layout.abi.worker_count, kMaxEthernetWriters);
    EXPECT_EQ(adm.layout.abi.risc_per_tile, 1u);
}

TEST(PacketInjectionErisc, WritersAreASortedNestedPrefix) {
    const std::vector<LogicalCoord> inactive = {{0, 5}, {0, 1}, {1, 0}, {0, 3}, {0, 2}};
    const WriterSelection one = select_ethernet_writers(inactive, 1);
    const WriterSelection four = select_ethernet_writers(inactive, 4);
    ASSERT_TRUE(one.ok()) << one.describe();
    ASSERT_TRUE(four.ok()) << four.describe();
    EXPECT_EQ(four.writers, (std::vector<LogicalCoord>{{0, 1}, {0, 2}, {0, 3}, {0, 5}}));
    EXPECT_EQ(one.writers.front(), four.writers.front());
    EXPECT_EQ(four.available, 5u);
}

// Too few cores is reported against the request, never narrowed to what the
// device has; "all" resolves to every core up to one device's Ethernet tiles.
TEST(PacketInjectionErisc, ShortfallIsReportedAndAllIsCapped) {
    std::vector<LogicalCoord> inactive;
    for (uint32_t y = 0; y < kMaxEthernetWriters + 2u; ++y) {
        inactive.push_back(LogicalCoord{0u, y});
    }
    const WriterSelection all = select_ethernet_writers(inactive, 0);
    ASSERT_TRUE(all.ok()) << all.describe();
    EXPECT_EQ(all.writers.size(), kMaxEthernetWriters);

    const std::vector<LogicalCoord> three(inactive.begin(), inactive.begin() + 3);
    const WriterSelection short8 = select_ethernet_writers(three, 8);
    EXPECT_EQ(short8.outcome, WriterSelection::Outcome::Shortfall);
    EXPECT_TRUE(short8.writers.empty());
    EXPECT_EQ(short8.requested, 8u);
    EXPECT_EQ(short8.available, 3u);

    EXPECT_EQ(select_ethernet_writers(inactive, kMaxEthernetWriters + 1u).outcome, WriterSelection::Outcome::BeyondCap);
}

TEST(PacketInjectionErisc, DuplicateCoreIsMalformedInput) {
    const WriterSelection sel = select_ethernet_writers({{0, 1}, {0, 2}, {0, 1}}, 2);
    EXPECT_EQ(sel.outcome, WriterSelection::Outcome::Duplicate);
    EXPECT_EQ(sel.duplicate, (LogicalCoord{0, 1}));
}
