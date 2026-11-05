// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Matching a plan's routers to the chips a reader found, one router per way it can turn out. No device.

#include <gtest/gtest.h>

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_device.hpp"
#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_manifest.hpp"
#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_targets.hpp"

namespace tt::tt_fabric::capture {
namespace {

PlannedRouter make_router(const std::string& path, uint32_t eth_chan, CoreXY virtual_core) {
    return PlannedRouter{.path = path, .eth_chan = eth_chan, .virtual_core = virtual_core};
}

// Three planned chips: M0/C0 (asic 0xa), whose routers each meet a different case, M0/C1 (0xb), which the reader
// didn't find, and M0/C2 (0xc), which the reader found as another architecture.
CapturePlan make_plan() {
    return CapturePlan{
        .arch = "blackhole",
        .chips =
            {{.path = "M0/C0",
              .asic_id = 0xa,
              .routers =
                  {make_router("M0/C0/E0", 0, {1, 1}),
                   make_router("M0/C0/E1", 1, {2, 1}),
                   make_router("M0/C0/E2", 5, {5, 1}),
                   make_router("M0/C0/E3", 3, {9, 9}),
                   make_router("M0/C0/E4", 4, {6, 1})}},
             {.path = "M0/C1", .asic_id = 0xb, .routers = {make_router("M0/C1/E0", 0, {1, 1})}},
             {.path = "M0/C2", .asic_id = 0xc, .routers = {make_router("M0/C2/E0", 0, {1, 1})}}},
    };
}

// Chip 0 (asic 0xa) has channels 0-4, with channel 1 and channel 2 at each other's manifest cores, and no channel 5.
// Discovery skipped its channel 4 core. Chip 1 (asic 0xc) is a Wormhole.
ClusterFacts make_facts() {
    return ClusterFacts{
        .cluster_id = "bh-glx-120-c02u14",
        .chips =
            {{.chip = 0,
              .asic_id = 0xa,
              .arch = "blackhole",
              .eth_cores_by_channel = {CoreXY{1, 1}, CoreXY{3, 1}, CoreXY{2, 1}, CoreXY{4, 1}, CoreXY{6, 1}}},
             {.chip = 1, .asic_id = 0xc, .arch = "wormhole_b0", .eth_cores_by_channel = {CoreXY{1, 1}}}},
        .skipped_cores = {{.asic_id = 0xa, .core = {6, 1}, .reason = "heartbeat stalled"}},
    };
}

TEST(CaptureTargets, PutsEachRouterInTheRightCase) {
    struct Expected {
        std::string router_path;
        CoreLocation location;
        std::optional<int32_t> chip;
        std::optional<CoreXY> reader_core;
        bool discovery_skipped;
        bool is_read;
    };
    const std::vector<Expected> expected = {
        {"M0/C0/E0", CoreLocation::FOUND, 0, CoreXY{1, 1}, false, true},
        // Channel 1 is at (3, 1), but the manifest's (2, 1) is channel 2's core, so the manifest's core is read.
        {"M0/C0/E1", CoreLocation::CORE_MISMATCH, 0, CoreXY{3, 1}, false, true},
        {"M0/C0/E2", CoreLocation::CHANNEL_NOT_FOUND, 0, std::nullopt, false, false},
        // Channel 3 is at (4, 1), and the manifest's (9, 9) is no Ethernet core of the chip.
        {"M0/C0/E3", CoreLocation::CORE_NOT_FOUND, 0, CoreXY{4, 1}, false, false},
        {"M0/C0/E4", CoreLocation::FOUND, 0, CoreXY{6, 1}, true, true},
        {"M0/C1/E0", CoreLocation::CHIP_MISSING, std::nullopt, std::nullopt, false, false},
        {"M0/C2/E0", CoreLocation::ARCH_MISMATCH, 1, CoreXY{1, 1}, false, false},
    };

    const CaptureTargets targets = resolve_capture_targets(make_plan(), make_facts());
    ASSERT_EQ(targets.cores.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        const CaptureTarget& target = targets.cores[i];
        SCOPED_TRACE(expected[i].router_path);
        EXPECT_EQ(target.router_path, expected[i].router_path);
        EXPECT_EQ(target.location, expected[i].location);
        EXPECT_EQ(target.chip, expected[i].chip);
        EXPECT_EQ(target.reader_core, expected[i].reader_core);
        EXPECT_EQ(target.discovery_skipped, expected[i].discovery_skipped);
        EXPECT_EQ(target.is_read(), expected[i].is_read);
    }
    EXPECT_EQ(targets.missing_chips, std::vector<std::string>{"M0/C1"});
}

}  // namespace
}  // namespace tt::tt_fabric::capture
