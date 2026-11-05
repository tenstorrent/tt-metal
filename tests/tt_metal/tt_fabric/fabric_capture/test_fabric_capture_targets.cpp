// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Matching a plan's chips and routers to what a reader found, one chip or router per way it can turn out. No device.

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

TEST(CaptureTargets, PutsEachChipAndRouterInTheRightCase) {
    const std::vector<ChipTarget> targets = resolve_capture_targets(make_plan(), make_facts());
    ASSERT_EQ(targets.size(), 3u);

    // M0/C0 was found, and each of its routers meets a different case.
    const ChipTarget& found = targets[0];
    EXPECT_EQ(found.location, ChipLocation::FOUND);
    EXPECT_EQ(found.chip, 0);
    struct Expected {
        std::string router_path;
        std::optional<CoreXY> reader_core;
        CoreLocation location;
        bool discovery_skipped;
        bool is_read;
    };
    const std::vector<Expected> expected = {
        {"M0/C0/E0", CoreXY{1, 1}, CoreLocation::FOUND, false, true},
        // Channel 1 is at (3, 1), but the manifest's (2, 1) is channel 2's core, so the manifest's core is read.
        {"M0/C0/E1", CoreXY{3, 1}, CoreLocation::CORE_MISMATCH, false, true},
        {"M0/C0/E2", std::nullopt, CoreLocation::CHANNEL_NOT_FOUND, false, false},
        // Channel 3 is at (4, 1), and the manifest's (9, 9) is no Ethernet core of the chip.
        {"M0/C0/E3", CoreXY{4, 1}, CoreLocation::CORE_NOT_FOUND, false, false},
        {"M0/C0/E4", CoreXY{6, 1}, CoreLocation::FOUND, true, true},
    };
    ASSERT_EQ(found.routers.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        const RouterTarget& router = found.routers[i];
        SCOPED_TRACE(expected[i].router_path);
        EXPECT_EQ(router.router.path, expected[i].router_path);
        EXPECT_EQ(router.reader_core, expected[i].reader_core);
        EXPECT_EQ(router.location, expected[i].location);
        EXPECT_EQ(router.discovery_skipped, expected[i].discovery_skipped);
        EXPECT_EQ(router.is_read(), expected[i].is_read);
    }

    // M0/C1 wasn't found, and M0/C2 is a Wormhole: their routers' cores aren't looked for, so none is read.
    const ChipTarget& missing = targets[1];
    EXPECT_EQ(missing.location, ChipLocation::MISSING);
    EXPECT_EQ(missing.chip, std::nullopt);
    const ChipTarget& other_arch = targets[2];
    EXPECT_EQ(other_arch.location, ChipLocation::ARCH_MISMATCH);
    EXPECT_EQ(other_arch.chip, 1);
    for (const ChipTarget* chip : {&missing, &other_arch}) {
        ASSERT_EQ(chip->routers.size(), 1u);
        EXPECT_EQ(chip->routers[0].location, std::nullopt);
        EXPECT_FALSE(chip->routers[0].is_read());
    }
}

}  // namespace
}  // namespace tt::tt_fabric::capture
