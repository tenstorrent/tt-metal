// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_targets.hpp"

#include <algorithm>
#include <utility>

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

// Where the reader found a planned chip: by its asic_id, then its architecture.
ChipLocation locate_chip(const CapturePlan& plan, const FoundChip* chip) {
    if (chip == nullptr) {
        return ChipLocation::MISSING;
    }
    return chip->arch == plan.arch ? ChipLocation::FOUND : ChipLocation::ARCH_MISMATCH;
}

// Where the reader found a router's core on its chip, checked in order: the channel, then the core.
CoreLocation locate_core(const FoundChip& chip, const PlannedRouter& router) {
    const std::optional<CoreXY> channel_core = find_channel_core(chip, router.eth_chan);
    if (!channel_core.has_value()) {
        return CoreLocation::CHANNEL_NOT_FOUND;
    }
    if (!is_eth_core(chip, router.virtual_core)) {
        return CoreLocation::CORE_NOT_FOUND;
    }
    return *channel_core == router.virtual_core ? CoreLocation::FOUND : CoreLocation::CORE_MISMATCH;
}

}  // namespace

std::vector<ChipTarget> resolve_capture_targets(const CapturePlan& plan, const ClusterFacts& facts) {
    std::vector<ChipTarget> targets;
    targets.reserve(plan.chips.size());
    for (const PlannedChip& planned_chip : plan.chips) {
        // Locate chip via asic_id from cluster facts
        const FoundChip* chip = find_chip(facts, planned_chip.asic_id);
        ChipTarget chip_target{
            .path = planned_chip.path,
            .asic_id = planned_chip.asic_id,
            .chip = chip == nullptr ? std::nullopt : std::optional(chip->chip),
            .location = locate_chip(plan, chip),
        };

        // For each router, find information about the core from the cluster facts
        for (const PlannedRouter& router : planned_chip.routers) {
            chip_target.routers.push_back(RouterTarget{
                .router = router,
                .reader_core = chip == nullptr ? std::nullopt : find_channel_core(*chip, router.eth_chan),
                .location = chip_target.location == ChipLocation::FOUND ? std::optional(locate_core(*chip, router))
                                                                        : std::nullopt,
                .discovery_skipped = was_skipped_by_discovery(facts, planned_chip.asic_id, router.virtual_core),
            });
        }
        targets.push_back(std::move(chip_target));
    }
    return targets;
}

}  // namespace tt::tt_fabric::capture
