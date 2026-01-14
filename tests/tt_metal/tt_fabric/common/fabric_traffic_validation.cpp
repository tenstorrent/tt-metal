// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <thread>
#include <tt-logger/tt-logger.hpp>

#include "fabric_traffic_validation.hpp"

namespace tt::tt_fabric::test_utils {

TelemetrySnapshot capture_telemetry_snapshot(
    ControlPlane& control_plane,
    MeshId mesh_id,
    size_t num_devices) {

    TelemetrySnapshot snapshot;

    for (size_t device_idx = 0; device_idx < num_devices; ++device_idx) {
        FabricNodeId node_id{mesh_id, static_cast<uint8_t>(device_idx)};

        try {
            auto telemetry_samples = control_plane.read_fabric_telemetry(node_id);

            for (const auto& sample : telemetry_samples) {
                // Handle dynamic_info being nullopt
                if (sample.snapshot.dynamic_info.has_value()) {
                    snapshot.words_sent_per_channel[node_id][sample.channel_id] =
                        sample.snapshot.dynamic_info->tx_bandwidth.words_sent;
                } else {
                    // Log warning but continue
                    log_warning(LogTest, "dynamic_info not available for node {} channel {}",
                        node_id, sample.channel_id);
                }
            }
        } catch (const std::exception& e) {
            log_warning(LogTest, "Failed to read telemetry for node {}: {}", node_id, e.what());
        }
    }

    return snapshot;
}

bool telemetry_changed(
    const TelemetrySnapshot& before,
    const TelemetrySnapshot& after) {

    for (const auto& [node_id, channels_after] : after.words_sent_per_channel) {
        auto it_before = before.words_sent_per_channel.find(node_id);
        if (it_before == before.words_sent_per_channel.end()) {
            continue;  // Skip nodes not in before snapshot - only compare existing data
        }

        for (const auto& [channel_id, words_sent_after] : channels_after) {
            auto chan_it = it_before->second.find(channel_id);
            if (chan_it == it_before->second.end()) {
                continue;
            }

            if (words_sent_after > chan_it->second) {
                return true;  // Traffic detected on this channel
            }
        }
    }

    return false;
}

bool validate_traffic_flowing(
    ControlPlane& control_plane,
    MeshId mesh_id,
    size_t num_devices,
    std::chrono::milliseconds sample_interval) {

    auto snapshot_before = capture_telemetry_snapshot(control_plane, mesh_id, num_devices);
    std::this_thread::sleep_for(sample_interval);
    auto snapshot_after = capture_telemetry_snapshot(control_plane, mesh_id, num_devices);

    return telemetry_changed(snapshot_before, snapshot_after);
}

bool validate_traffic_stopped(
    ControlPlane& control_plane,
    MeshId mesh_id,
    size_t num_devices,
    std::chrono::milliseconds sample_interval) {

    auto snapshot_before = capture_telemetry_snapshot(control_plane, mesh_id, num_devices);
    std::this_thread::sleep_for(sample_interval);
    auto snapshot_after = capture_telemetry_snapshot(control_plane, mesh_id, num_devices);

    // Return true if NO channels changed (traffic stopped)
    return !telemetry_changed(snapshot_before, snapshot_after);
}

}  // namespace tt::tt_fabric::test_utils
