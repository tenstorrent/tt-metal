// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/fabric_telemetry.hpp>

namespace tt::tt_fabric {
class FabricNodeId;
}

namespace tt::tt_metal {

/**
 * Represents a telemetry sample captured from a single router direction on a fabric node.
 */
struct FabricTelemetrySample {
    std::uint8_t channel_id;
    CoreCoord logical_eth_core;
    tt::tt_fabric::FabricTelemetrySnapshot snapshot;
};

/**
 * @brief Read telemetry snapshots for all active ethernet channels on the specified fabric node.
 *
 * The snapshots are converted into the public API structures so callers only need the Metalium API headers.
 */
std::vector<FabricTelemetrySample> ReadFabricTelemetrySnapshots(const tt::tt_fabric::FabricNodeId& fabric_node_id);

/**
 * @brief Read telemetry snapshot for a specific ethernet channel on the specified fabric node.
 *
 * @return std::nullopt if the channel is not active on the node.
 */
std::optional<FabricTelemetrySample> ReadFabricTelemetrySnapshot(
    const tt::tt_fabric::FabricNodeId& fabric_node_id, std::uint8_t channel_id);

}  // namespace tt::tt_metal
