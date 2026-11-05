// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_targets.hpp"

#include <algorithm>

namespace tt::tt_fabric::capture {

namespace {

// Finds a chip in the cluster facts by its asic_id.
const FoundChip* find_chip(const ClusterFacts& facts, uint64_t asic_id) {
    const auto it = std::ranges::find(facts.chips, asic_id, &FoundChip::asic_id);
    return it == facts.chips.end() ? nullptr : &*it;
}

// Finds a channel's core in a chip by its channel index.
std::optional<CoreXY> find_channel_core(const FoundChip& chip, uint32_t eth_chan) {
    return eth_chan < chip.eth_cores_by_channel.size() ? chip.eth_cores_by_channel[eth_chan] : std::nullopt;
}

// Checks if a core is an Ethernet core of a chip by its channel index.
bool is_eth_core(const FoundChip& chip, const CoreXY& core) {
    return std::ranges::find(chip.eth_cores_by_channel, std::optional(core)) != chip.eth_cores_by_channel.end();
}

// Checks if a core was skipped by discovery.
bool was_skipped_by_discovery(const ClusterFacts& facts, uint64_t asic_id, const CoreXY& core) {
    return std::ranges::any_of(facts.skipped_cores, [&](const DiscoverySkippedCore& skipped) {
        return skipped.asic_id == asic_id && skipped.core == core;
    });
}

// Where the reader found the router's core, checked in order: the chip, its architecture, the channel, then the core.
CoreLocation locate_core(const CapturePlan& plan, const FoundChip* chip, const PlannedRouter& router) {
    if (chip == nullptr) {
        return CoreLocation::CHIP_MISSING;
    }
    if (chip->arch != plan.arch) {
        return CoreLocation::ARCH_MISMATCH;
    }
    const std::optional<CoreXY> channel_core = find_channel_core(*chip, router.eth_chan);
    if (!channel_core.has_value()) {
        return CoreLocation::CHANNEL_NOT_FOUND;
    }
    if (!is_eth_core(*chip, router.virtual_core)) {
        return CoreLocation::CORE_NOT_FOUND;
    }
    return *channel_core == router.virtual_core ? CoreLocation::FOUND : CoreLocation::CORE_MISMATCH;
}

}  // namespace

CaptureTargets resolve_capture_targets(const CapturePlan& plan, const ClusterFacts& facts) {
    CaptureTargets targets;
    for (const PlannedChip& planned_chip : plan.chips) {
        // Locate chip via asic_id from cluster facts
        const FoundChip* chip = find_chip(facts, planned_chip.asic_id);
        if (chip == nullptr) {
            targets.missing_chips.push_back(planned_chip.path);
        }

        // For each router, find information about the core from the cluster facts
        for (const PlannedRouter& router : planned_chip.routers) {
            targets.cores.push_back(CaptureTarget{
                .router = router,
                .asic_id = planned_chip.asic_id,
                .chip = chip == nullptr ? std::nullopt : std::optional(chip->chip),
                .reader_core = chip == nullptr ? std::nullopt : find_channel_core(*chip, router.eth_chan),
                .location = locate_core(plan, chip, router),
                .discovery_skipped = was_skipped_by_discovery(facts, planned_chip.asic_id, router.virtual_core),
            });
        }
    }
    return targets;
}

}  // namespace tt::tt_fabric::capture
