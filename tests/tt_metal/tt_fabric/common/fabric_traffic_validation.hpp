// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <map>
#include <cstdint>

#include "tt_metal/fabric/control_plane.hpp"
#include "tests/tt_metal/tt_fabric/common/fabric_traffic_generator_defs.hpp"

namespace tt::tt_fabric::test_utils {

// Helper: Capture telemetry snapshot across all devices
struct TelemetrySnapshot {
    std::map<FabricNodeId, std::map<chan_id_t, uint64_t>> words_sent_per_channel;
};

// Capture current telemetry snapshot
TelemetrySnapshot capture_telemetry_snapshot(
    ControlPlane& control_plane,
    MeshId mesh_id,
    size_t num_devices);

// Helper: Compare two snapshots
// Returns true if any channel's words_sent increased
bool telemetry_changed(
    const TelemetrySnapshot& before,
    const TelemetrySnapshot& after);

// FR-3: Validate that traffic is flowing on at least one channel
// Returns true if words_sent increased between two samples
bool validate_traffic_flowing(
    ControlPlane& control_plane,
    MeshId mesh_id,
    size_t num_devices,
    std::chrono::milliseconds sample_interval = DEFAULT_TRAFFIC_SAMPLE_INTERVAL);

// FR-7: Validate that traffic has stopped on all channels
// Returns true if words_sent did not increase between two samples
bool validate_traffic_stopped(
    ControlPlane& control_plane,
    MeshId mesh_id,
    size_t num_devices,
    std::chrono::milliseconds sample_interval = DEFAULT_TRAFFIC_SAMPLE_INTERVAL);

}  // namespace tt::tt_fabric::test_utils
