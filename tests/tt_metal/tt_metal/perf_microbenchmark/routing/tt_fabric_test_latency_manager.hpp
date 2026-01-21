// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <unordered_map>
#include <vector>
#include <map>
#include <optional>
#include <string>
#include <cmath>
#include <tt-logger/tt-logger.hpp>

#include "tt_fabric_test_common_types.hpp"
#include "tt_fabric_test_common.hpp"
#include "tt_fabric_test_interfaces.hpp"
#include "tt_fabric_test_memory_map.hpp"
#include "tt_fabric_test_device_setup.hpp"
#include "tt_fabric_test_traffic.hpp"
#include "tt_fabric_test_constants.hpp"
#include "tt_fabric_test_results.hpp"

using TestFixture = tt::tt_fabric::fabric_tests::TestFixture;
using TestDevice = tt::tt_fabric::fabric_tests::TestDevice;
using TestConfig = tt::tt_fabric::fabric_tests::TestConfig;
using SenderMemoryMap = tt::tt_fabric::fabric_tests::SenderMemoryMap;
using TrafficPatternConfig = tt::tt_fabric::fabric_tests::TrafficPatternConfig;
using TrafficParameters = tt::tt_fabric::fabric_tests::TrafficParameters;
using TestTrafficSenderConfig = tt::tt_fabric::fabric_tests::TestTrafficSenderConfig;
using TestTrafficReceiverConfig = tt::tt_fabric::fabric_tests::TestTrafficReceiverConfig;
using PerformanceTestMode = tt::tt_fabric::fabric_tests::PerformanceTestMode;
using LatencyResult = tt::tt_fabric::fabric_tests::LatencyResult;
using GoldenLatencyEntry = tt::tt_fabric::fabric_tests::GoldenLatencyEntry;
using LatencyComparisonResult = tt::tt_fabric::fabric_tests::LatencyComparisonResult;
using MeshCoordinate = tt::tt_metal::distributed::MeshCoordinate;
using CoreCoord = tt::tt_metal::CoreCoord;
using FabricNodeId = tt::tt_fabric::FabricNodeId;
using MeshId = tt::tt_fabric::MeshId;
using NocSendType = tt::tt_fabric::NocSendType;
using tt::tt_fabric::fabric_tests::CI_ARTIFACTS_DIR;
using tt::tt_fabric::fabric_tests::OUTPUT_DIR;

// Manages latency test setup, execution, result collection, CSV generation, and golden comparison.
class LatencyTestManager {
public:
    struct LatencyWorkerLocation {
        TestDevice* device = nullptr;
        MeshCoordinate mesh_coord{0, 0};
        CoreCoord core;
        FabricNodeId node_id{MeshId{0}, 0};
    };

    LatencyTestManager(TestFixture& fixture, SenderMemoryMap& sender_memory_map);

    void setup_latency_test_mode(const TestConfig& config);
    void setup_latency_test_workers(TestConfig& config, std::unordered_map<MeshCoordinate, TestDevice>& test_devices);
    void create_latency_kernels_for_device(
        TestDevice& test_device, std::unordered_map<MeshCoordinate, TestDevice>& test_devices);

    LatencyWorkerLocation get_latency_sender_location(std::unordered_map<MeshCoordinate, TestDevice>& test_devices);
    LatencyWorkerLocation get_latency_receiver_location(std::unordered_map<MeshCoordinate, TestDevice>& test_devices);

    void collect_latency_results(std::unordered_map<MeshCoordinate, TestDevice>& test_devices);
    void report_latency_results(const TestConfig& config, std::unordered_map<MeshCoordinate, TestDevice>& test_devices);

    void initialize_latency_results_csv_file();
    void generate_latency_results_csv();
    std::string get_golden_latency_csv_filename();
    bool load_golden_latency_csv();
    void compare_latency_results_with_golden();
    void generate_latency_summary();
    void setup_ci_artifacts();

    bool has_failures() const { return has_failures_; }
    std::vector<std::string> get_failed_tests() const { return all_failed_latency_tests_; }
    const std::vector<LatencyResult>& get_latency_results() const { return latency_results_; }

    void reset_state();

private:
    template <typename GetWorkersMapFunc>
    LatencyWorkerLocation find_latency_worker_device(
        std::unordered_map<MeshCoordinate, TestDevice>& test_devices,
        GetWorkersMapFunc get_workers_map,
        const std::string& worker_type);

    template <typename CompResultType, typename GoldenIterType>
    void populate_comparison_tolerance_and_status(
        CompResultType& comp_result,
        GoldenIterType golden_it,
        GoldenIterType golden_end,
        double golden_tolerance_default = 1.0);

    std::ofstream init_diff_csv_file(
        std::filesystem::path& diff_csv_path, const std::string& csv_header, const std::string& test_type);

    void validate_against_golden();

    TestFixture& fixture_;
    SenderMemoryMap& sender_memory_map_;

    std::vector<LatencyResult> latency_results_;
    std::vector<GoldenLatencyEntry> golden_latency_entries_;
    std::vector<LatencyComparisonResult> latency_comparison_results_;
    std::vector<std::string> all_failed_latency_tests_;

    std::filesystem::path latency_csv_file_path_;
    std::filesystem::path latency_diff_csv_file_path_;

    bool has_failures_ = false;
};

// Template definitions
template <typename GetWorkersMapFunc>
LatencyTestManager::LatencyWorkerLocation LatencyTestManager::find_latency_worker_device(
    std::unordered_map<MeshCoordinate, TestDevice>& test_devices,
    GetWorkersMapFunc get_workers_map,
    const std::string& worker_type) {
    LatencyWorkerLocation info;
    for (auto& [coord, device] : test_devices) {
        const auto& workers_map = get_workers_map(device);
        if (!workers_map.empty()) {
            info.device = &device;
            info.mesh_coord = coord;
            info.core = workers_map.begin()->first;
            info.node_id = device.get_node_id();
            break;
        }
    }
    TT_FATAL(info.device != nullptr, "Could not find latency {} device", worker_type);
    return info;
}

template <typename CompResultType, typename GoldenIterType>
void LatencyTestManager::populate_comparison_tolerance_and_status(
    CompResultType& comp_result, GoldenIterType golden_it, GoldenIterType golden_end, double golden_tolerance_default) {
    double test_tolerance = golden_tolerance_default;
    if (golden_it != golden_end) {
        test_tolerance = golden_it->tolerance_percent;
        comp_result.within_tolerance = std::abs(comp_result.difference_percent()) <= test_tolerance;
        comp_result.status = comp_result.within_tolerance ? "PASS" : "FAIL";
    } else {
        log_warning(tt::LogTest, "Golden entry not found for test {}", comp_result.test_name);
        comp_result.within_tolerance = false;
        comp_result.status = "NO_GOLDEN";
    }
}
