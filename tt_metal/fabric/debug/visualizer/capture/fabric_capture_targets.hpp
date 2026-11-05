// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_utils.hpp"
#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_device.hpp"
#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_manifest.hpp"

namespace tt::tt_fabric::capture {

// Where the reader found a planned chip, against the manifest.
enum class ChipLocation {
    // A chip the reader found has the manifest's asic_id, and it's the manifest's architecture.
    FOUND,
    // No chip the reader found has the manifest's asic_id. None of its routers is read.
    MISSING,
    // The chip isn't the manifest's architecture, so the manifest's addresses don't apply. None of its routers is read.
    ARCH_MISMATCH,
};

// Where the reader found a router's core, on a chip it found.
enum class CoreLocation {
    // The reader puts the router's channel at the manifest's core.
    FOUND,
    // The reader puts the channel at another core. The manifest's core, where the run launched the router, is an
    // Ethernet core of the chip, so it's read.
    CORE_MISMATCH,
    // The chip has no Ethernet core for the router's channel. Not read.
    CHANNEL_NOT_FOUND,
    // The manifest's core isn't an Ethernet core of the chip. Not read: a NOC read of a coordinate that isn't a core
    // can hang the NOC.
    CORE_NOT_FOUND,
};

// A router, matched to what the reader found on its chip.
struct RouterTarget {
    // The router as the plan has it: its path, channel, manifest core and stream registers, which is what a read of
    // its core needs.
    PlannedRouter router;
    // The reader's core for the router's channel, when the chip was found and has one.
    std::optional<CoreXY> reader_core;
    // Null when the chip wasn't found, or isn't the manifest's architecture: then the core isn't looked for.
    std::optional<CoreLocation> location;
    // Topology discovery skipped the core. It's still read.
    bool discovery_skipped = false;

    bool operator==(const RouterTarget&) const = default;
    // Whether capture reads the core, at the manifest's core.
    bool is_read() const { return location == CoreLocation::FOUND || location == CoreLocation::CORE_MISMATCH; }
};

// A planned chip, matched to a chip the reader found, with its routers.
struct ChipTarget {
    // The chip's path, such as "M0/C7".
    std::string path;
    uint64_t asic_id = 0;
    // The reader's id for the chip, when it found the chip.
    std::optional<int32_t> chip;
    ChipLocation location = ChipLocation::FOUND;
    // One per planned router, in the plan's order.
    std::vector<RouterTarget> routers;

    bool operator==(const ChipTarget&) const = default;
};

// Uses ClusterFacts to match each planned chip (from manifest) to a chip discovered by cluster discovery. Handles
// matching routers as well, and identifies discrepancies between the manifest and the cluster facts.
std::vector<ChipTarget> resolve_capture_targets(const CapturePlan& plan, const ClusterFacts& facts);

}  // namespace tt::tt_fabric::capture
