// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>
#include <vector>
#include <map>
#include <optional>

#include "tt_fabric_test_common_types.hpp"
#include "tt_fabric_test_common.hpp"
#include "tt_fabric_test_interfaces.hpp"
#include "tt_fabric_test_memory_map.hpp"
#include "tt_fabric_test_traffic.hpp"
#include "tt_fabric_test_results.hpp"
#include "tt_fabric_test_device_setup.hpp"
#include "tt_fabric_test_constants.hpp"

using TestFixture = tt::tt_fabric::fabric_tests::TestFixture;
using TestDevice = tt::tt_fabric::fabric_tests::TestDevice;
using TestConfig = tt::tt_fabric::fabric_tests::TestConfig;
using MeshCoordinate = tt::tt_metal::distributed::MeshCoordinate;
using CoreCoord = tt::tt_metal::CoreCoord;
using FabricNodeId = tt::tt_fabric::FabricNodeId;
using RoutingDirection = tt::tt_fabric::RoutingDirection;
using SenderMemoryMap = tt::tt_fabric::fabric_tests::SenderMemoryMap;
using TrafficPatternConfig = tt::tt_fabric::fabric_tests::TrafficPatternConfig;
using BandwidthResult = tt::tt_fabric::fabric_tests::BandwidthResult;
using BandwidthResultSummary = tt::tt_fabric::fabric_tests::BandwidthResultSummary;
using IDeviceInfoProvider = tt::tt_fabric::fabric_tests::IDeviceInfoProvider;
using IRouteManager = tt::tt_fabric::fabric_tests::IRouteManager;
using TestTrafficSenderConfig = tt::tt_fabric::fabric_tests::TestTrafficSenderConfig;

// Calculates bandwidth metrics for a single test iteration.
class BandwidthProfiler {
public:
    BandwidthProfiler(IDeviceInfoProvider& device_info, IRouteManager& route_manager, TestFixture& fixture);

    // Pass memory map per call (policies may change between tests).
    void profile_results(
        const TestConfig& config,
        const std::unordered_map<MeshCoordinate, TestDevice>& test_devices,
        const SenderMemoryMap& sender_memory_map);

    const BandwidthResult& get_latest_result() const { return latest_result_; }
    const BandwidthResultSummary& get_latest_summary() const { return latest_summary_; }
    const std::vector<BandwidthResult>& get_latest_results() const { return latest_results_; }

    // Inject telemetry (must be called before reading latest_result()).
    void set_telemetry_bandwidth(double min, double avg, double max);

    // Clears per-test computation state.
    void reset();

private:
    [[maybe_unused]] IDeviceInfoProvider& device_info_;
    IRouteManager& route_manager_;
    TestFixture& fixture_;

    BandwidthResult latest_result_{};
    BandwidthResultSummary latest_summary_{};
    std::vector<BandwidthResult> latest_results_;

    // Internal scratch state (per test)
    std::map<FabricNodeId, std::map<RoutingDirection, uint32_t>> outgoing_traffic_;
    std::map<FabricNodeId, std::map<RoutingDirection, std::map<uint32_t, uint64_t>>> device_direction_cycles_;
    std::map<FabricNodeId, std::map<CoreCoord, uint64_t>> device_core_cycles_;
    std::optional<double> telemetry_bw_min_;
    std::optional<double> telemetry_bw_avg_;
    std::optional<double> telemetry_bw_max_;

    // Helpers moved from TestContext
    std::map<FabricNodeId, std::map<RoutingDirection, uint32_t>> calculate_outgoing_traffics_through_device_boundaries(
        const std::unordered_map<MeshCoordinate, TestDevice>&);

    void trace_traffic_path(const FabricNodeId& src_node_id, const TestTrafficSenderConfig& config);
    void trace_ring_traffic_path(const FabricNodeId& src_node_id, const TestTrafficSenderConfig& config);
    void trace_line_or_mesh_traffic_path(const FabricNodeId& src_node_id, const TestTrafficSenderConfig& config);

    void read_performance_results(
        const std::unordered_map<MeshCoordinate, TestDevice>& test_devices, const SenderMemoryMap& sender_memory_map);
    void convert_core_cycles_to_direction_cycles(const std::unordered_map<MeshCoordinate, TestDevice>& test_devices);
    void calculate_bandwidth(
        const TestConfig& config, const std::unordered_map<MeshCoordinate, TestDevice>& test_devices);
};
