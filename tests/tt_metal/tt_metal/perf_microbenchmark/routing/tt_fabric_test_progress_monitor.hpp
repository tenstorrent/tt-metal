// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>

#include "tt_fabric_test_interfaces.hpp"
#include "tt_fabric_test_common.hpp"                // For MeshCoordinate
#include <tt-metalium/routing_table_generator.hpp>  // For FabricNodeId
#include <tt-metalium/mesh_graph.hpp>               // For MeshId

// Forward declarations
class TestContext;

namespace tt::tt_fabric::fabric_tests {

struct TestDevice;

// Progress monitoring configuration
struct ProgressMonitorConfig {
    bool enabled = false;
    uint32_t poll_interval_seconds = 2;    // How often to poll progress
    uint32_t hung_threshold_seconds = 30;  // When to warn about hung devices
    bool verbose = false;                  // Show per-device progress vs summary
};

// Progress data for a single device
struct DeviceProgress {
    tt::tt_fabric::FabricNodeId device_id{tt::tt_fabric::MeshId{0}, 0};  // Default: mesh 0, chip 0
    uint64_t current_packets = 0;                                        // Total packets sent so far
    uint64_t total_packets = 0;                                          // Total packets to send
    uint32_t num_senders = 0;                                            // Number of sender cores
    uint32_t num_receivers = 0;  // Number of receiver cores (not currently monitored)
};

// Tracks state for hung detection
struct DeviceState {
    uint64_t last_packet_count = 0;
    std::chrono::steady_clock::time_point last_progress_time;
    bool warned = false;
};

// Progress monitor - polls devices and displays progress during test execution
class TestProgressMonitor {
public:
    TestProgressMonitor(::TestContext* ctx, const ProgressMonitorConfig& config);
    ~TestProgressMonitor();

    // Disable copy/move
    TestProgressMonitor(const TestProgressMonitor&) = delete;
    TestProgressMonitor& operator=(const TestProgressMonitor&) = delete;
    TestProgressMonitor(TestProgressMonitor&&) = delete;
    TestProgressMonitor& operator=(TestProgressMonitor&&) = delete;

    // Poll until programs complete (runs in calling thread)
    void poll_until_complete();

private:
    // Poll all devices and collect progress
    std::unordered_map<tt::tt_fabric::FabricNodeId, DeviceProgress> poll_devices();

    // Poll a single device's senders
    DeviceProgress poll_device_senders(const MeshCoordinate& coord, const TestDevice& test_device);

    // Check for hung devices and display warnings
    void check_for_hung_devices(const std::unordered_map<tt::tt_fabric::FabricNodeId, DeviceProgress>& progress);

    // Display progress (summary or verbose based on config)
    void display_progress(
        const std::unordered_map<tt::tt_fabric::FabricNodeId, DeviceProgress>& progress,
        std::chrono::duration<double> elapsed_since_last_poll);

    // Display progress in summary format (single line)
    void display_summary_progress(
        const std::unordered_map<tt::tt_fabric::FabricNodeId, DeviceProgress>& progress,
        std::chrono::duration<double> elapsed);

    // Display progress in verbose format (per-device with bars)
    void display_verbose_progress(
        const std::unordered_map<tt::tt_fabric::FabricNodeId, DeviceProgress>& progress,
        std::chrono::duration<double> elapsed);

    // Formatting helpers
    std::string format_count(uint64_t count) const;
    std::string format_throughput(double packets_per_second) const;
    std::string format_duration(double seconds) const;
    std::string format_progress_bar(double percentage, uint32_t width) const;

    // Hung detection
    bool is_device_hung(tt::tt_fabric::FabricNodeId device_id, uint64_t current_packets);

    // ETA calculation
    std::optional<double> estimate_eta(uint64_t current_total, uint64_t target_total, double throughput) const;

    // Context and configuration
    ::TestContext* ctx_;
    ProgressMonitorConfig config_;

    // Timing
    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::time_point last_poll_time_;

    // State for throughput calculation
    uint64_t last_total_packets_ = 0;

    // Hung detection state
    std::unordered_map<tt::tt_fabric::FabricNodeId, DeviceState> device_states_;
    std::chrono::seconds hung_threshold_;

    // Display state (for verbose mode cursor management)
    bool first_display_ = true;
};

}  // namespace tt::tt_fabric::fabric_tests
