// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include "tt_fabric_test_results.hpp"
#include "tt_fabric_test_common_types.hpp"
#include "tt_fabric_test_interfaces.hpp"
#include "tt_fabric_test_memory_map.hpp"
#include "tt_fabric_test_device_setup.hpp"
#include "tt_fabric_test_traffic.hpp"
#include "tt_fabric_test_constants.hpp"
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>

using TestFixture = tt::tt_fabric::fabric_tests::TestFixture;
using Topology = tt::tt_fabric::Topology;
using NocSendType = tt::tt_fabric::NocSendType;
using RoutingDirection = tt::tt_fabric::RoutingDirection;
using TestDevice = tt::tt_fabric::fabric_tests::TestDevice;
using TestConfig = tt::tt_fabric::fabric_tests::TestConfig;
using SenderMemoryMap = tt::tt_fabric::fabric_tests::SenderMemoryMap;
using TrafficPatternConfig = tt::tt_fabric::fabric_tests::TrafficPatternConfig;
using TrafficParameters = tt::tt_fabric::fabric_tests::TrafficParameters;
using TestTrafficSenderConfig = tt::tt_fabric::fabric_tests::TestTrafficSenderConfig;
using TestTrafficReceiverConfig = tt::tt_fabric::fabric_tests::TestTrafficReceiverConfig;
using PerformanceTestMode = tt::tt_fabric::fabric_tests::PerformanceTestMode;
using MeshCoordinate = tt::tt_metal::distributed::MeshCoordinate;
using CoreCoord = tt::tt_metal::CoreCoord;
using FabricNodeId = tt::tt_fabric::FabricNodeId;
using MeshId = tt::tt_fabric::MeshId;
using tt::tt_fabric::fabric_tests::CI_ARTIFACTS_DIR;
using tt::tt_fabric::fabric_tests::OUTPUT_DIR;

namespace tt::tt_fabric::fabric_tests {

// Latency measurement result structure
struct LatencyResult {
    std::string test_name;
    std::string ftype;
    std::string ntype;
    std::string topology;
    uint32_t num_devices{};
    uint32_t num_links{};
    uint32_t num_samples{};
    uint32_t payload_size{};

    // Statistics for net latency (raw - responder) - MOST IMPORTANT METRIC
    double net_min_ns{};
    double net_max_ns{};
    double net_avg_ns{};
    double net_p99_ns{};

    // Statistics for responder processing time
    double responder_min_ns{};
    double responder_max_ns{};
    double responder_avg_ns{};
    double responder_p99_ns{};

    // Statistics for raw latency (round-trip time)
    double raw_min_ns{};
    double raw_max_ns{};
    double raw_avg_ns{};
    double raw_p99_ns{};

    // Statistics for per-hop latency (net latency / num_hops)
    double per_hop_min_ns{};
    double per_hop_max_ns{};
    double per_hop_avg_ns{};
    double per_hop_p99_ns{};
};

struct LatencyResultSummary {
    std::string test_name;
    std::string ftype;
    std::string ntype;
    std::string topology;
    uint32_t num_devices{};
    uint32_t num_links{};
    uint32_t num_samples{};
    uint32_t payload_size{};

    // Statistics for net latency (raw - responder) - MOST IMPORTANT METRIC
    double net_min_ns{};
    double net_max_ns{};
    double net_avg_ns{};
    double net_p99_ns{};

    // Statistics for responder processing time
    double responder_min_ns{};
    double responder_max_ns{};
    double responder_avg_ns{};
    double responder_p99_ns{};

    // Statistics for raw latency (round-trip time)
    double raw_min_ns{};
    double raw_max_ns{};
    double raw_avg_ns{};
    double raw_p99_ns{};

    // Statistics for per-hop latency (net latency / num_hops)
    double per_hop_min_ns{};
    double per_hop_max_ns{};
    double per_hop_avg_ns{};
    double per_hop_p99_ns{};

    // Optional fields for database upload CSV
    std::optional<std::string> file_name;
    std::optional<std::string> machine_type;
    std::optional<std::string> test_ts;
};

// Golden latency comparison structures
// Extends LatencyResult with tolerance information from golden files
struct GoldenLatencyEntry : LatencyResult {
    double tolerance_percent{};  // Per-test tolerance percentage from golden CSV
};

struct LatencyComparisonResult {
    double speedup() const { return golden_per_hop_avg_ns / current_per_hop_avg_ns; }  // Lower is better for latency
    double difference_percent() const {
        return ((current_per_hop_avg_ns - golden_per_hop_avg_ns) / golden_per_hop_avg_ns) * 100.0;
    }

    std::string test_name;
    std::string ftype;
    std::string ntype;
    std::string topology;
    uint32_t num_devices{};
    uint32_t num_links{};
    uint32_t num_samples{};
    uint32_t payload_size{};

    // Current vs golden for per-hop latency average (primary comparison metric)
    double current_per_hop_avg_ns{};
    double golden_per_hop_avg_ns{};

    bool within_tolerance{};
    std::string status;
};

// Manages latency test setup, execution, result collection, CSV generation, and golden comparison.
class LatencyResultsManager : public ResultsManager<LatencyResult, LatencyResultSummary> {
public:
    struct LatencyWorkerLocation {
        TestDevice* device = nullptr;
        MeshCoordinate mesh_coord{0, 0};
        CoreCoord core;
        FabricNodeId node_id{MeshId{0}, 0};
    };

    LatencyResultsManager(TestFixture& fixture, SenderMemoryMap& sender_memory_map);

    void setup_latency_test_mode(const TestConfig& config);
    void setup_latency_test_workers(TestConfig& config, std::unordered_map<MeshCoordinate, TestDevice>& test_devices);
    void create_latency_kernels_for_device(
        TestDevice& test_device, std::unordered_map<MeshCoordinate, TestDevice>& test_devices);

    LatencyWorkerLocation get_latency_sender_location(std::unordered_map<MeshCoordinate, TestDevice>& test_devices);
    LatencyWorkerLocation get_latency_receiver_location(std::unordered_map<MeshCoordinate, TestDevice>& test_devices);

    void collect_latency_results(std::unordered_map<MeshCoordinate, TestDevice>& test_devices);
    void report_latency_results(const TestConfig& config, std::unordered_map<MeshCoordinate, TestDevice>& test_devices);

    void generate_summary() override;
    void initialize_results_csv_file(bool telemetry_enabled_) override;
    void load_golden_csv() override;
    void write_summary_csv_to_file(const std::filesystem::path& csv_path, bool include_upload_columns) override;
    void append_to_csv(const TestConfig& config, const LatencyResult& result);
    void compare_latency_results_with_golden();
    void generate_latency_summary();
    void setup_ci_artifacts();

    bool has_failures() const { return has_failures_; }
    std::vector<std::string> get_failed_tests() const { return all_failed_latency_tests_; }
    const std::vector<LatencyResult>& get_latency_results() const { return results_; }

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

    void validate_against_golden();

    TestFixture& fixture_;
    SenderMemoryMap& sender_memory_map_;

    std::vector<GoldenLatencyEntry> golden_latency_entries_;
    std::vector<LatencyComparisonResult> latency_comparison_results_;
    std::vector<std::string> all_failed_latency_tests_;

    bool has_failures_ = false;
};

}  // namespace tt::tt_fabric::fabric_tests
