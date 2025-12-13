// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/experimental/fabric/fabric_telemetry.hpp>
#include <tt-metalium/experimental/fabric/routing_table_generator.hpp>
#include <hostdevcommon/fabric_common.h>

namespace tt::tt_fabric {

/**
 * @brief Telemetry sample captured from a fabric router Ethernet channel.
 */
struct FabricTelemetrySample {
    tt::tt_fabric::FabricNodeId fabric_node_id{tt::tt_fabric::MeshId{0}, 0};
    tt::tt_fabric::chan_id_t channel_id = 0;
    tt::tt_fabric::FabricTelemetrySnapshot snapshot;
};

/**
 * @brief Read telemetry snapshots for all active Ethernet channels on the requested fabric node.
 *
 * The returned samples contain both the static router metadata and (when enabled in firmware) the dynamic counters that
 * are maintained by the router.  Memory access and HAL-dependent conversions are handled internally so that
 * applications can rely entirely on the public `tt::tt_fabric` types.
 *
 * @param fabric_node_id Logical fabric node identifier (mesh, chip) to query.
 * @return Telemetry samples, one per active Ethernet channel on the node. Returns an empty vector if the node has no
 *         active channels.
 */
[[nodiscard]] std::vector<FabricTelemetrySample> read_fabric_telemetry(
    const tt::tt_fabric::FabricNodeId& fabric_node_id);

}  // namespace tt::tt_fabric
