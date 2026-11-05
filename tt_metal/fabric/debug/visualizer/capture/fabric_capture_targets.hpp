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

// Where the reader found a router's core, against the manifest.
enum class CoreLocation {
    // The reader puts the router's channel at the manifest's core.
    FOUND,
    // The reader puts the channel at another core. The manifest's core, where the run launched the router, is an
    // Ethernet core of the chip, so it's read.
    CORE_MISMATCH,
    // No chip the reader found has the manifest's asic_id. Not read.
    CHIP_MISSING,
    // The chip isn't the manifest's architecture. Not read.
    ARCH_MISMATCH,
    // The chip has no Ethernet core for the router's channel. Not read.
    CHANNEL_NOT_FOUND,
    // The manifest's core isn't an Ethernet core of the chip. Not read: a NOC read of a coordinate that isn't a core
    // can hang the NOC.
    CORE_NOT_FOUND,
};

// A router's core, matched to what the reader found.
struct CaptureTarget {
    // The router as the plan has it: its path, channel, manifest core and stream registers, which is what a read of
    // its core needs.
    PlannedRouter router;
    uint64_t asic_id = 0;
    // The reader's id for the chip, when it found the chip.
    std::optional<int32_t> chip;
    // The reader's core for the channel, when the chip has one.
    std::optional<CoreXY> reader_core;
    CoreLocation location = CoreLocation::FOUND;
    // Topology discovery skipped the core. It's still read.
    bool discovery_skipped = false;

    bool operator==(const CaptureTarget&) const = default;
    // Whether capture reads the core.
    bool is_read() const { return location == CoreLocation::FOUND || location == CoreLocation::CORE_MISMATCH; }
};

// Every router a plan names, matched to what the reader found.
struct CaptureTargets {
    // One per planned router, in the plan's order.
    std::vector<CaptureTarget> cores;
    // The paths of the planned chips the reader didn't find.
    std::vector<std::string> missing_chips;
};

// Uses ClusterFacts to match each planned chip (from manifest) to a chip discovered by cluster discovery. Handles
// matching routers as well, and identifies discrepancies between the manifest and the cluster facts.
CaptureTargets resolve_capture_targets(const CapturePlan& plan, const ClusterFacts& facts);

}  // namespace tt::tt_fabric::capture
