// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <fstream>
#include <optional>
#include <functional>
#include "tt_fabric_test_config.hpp"
#include <enchantum/enchantum.hpp>

#include "tt_metal/fabric/fabric_edm_packet_header.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>

using Topology = tt::tt_fabric::Topology;
using NocSendType = tt::tt_fabric::NocSendType;
using RoutingDirection = tt::tt_fabric::RoutingDirection;

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
using MeshCoordinate = tt::tt_metal::distributed::MeshCoordinate;
using CoreCoord = tt::tt_metal::CoreCoord;
using FabricNodeId = tt::tt_fabric::FabricNodeId;
using MeshId = tt::tt_fabric::MeshId;
using NocSendType = tt::tt_fabric::NocSendType;
using tt::tt_fabric::fabric_tests::CI_ARTIFACTS_DIR;
using tt::tt_fabric::fabric_tests::OUTPUT_DIR;
namespace tt::tt_fabric::fabric_tests {

//=============================ENUMS==========================

// Measurement type
enum class PerformanceMetric { Bandwidth = 0, Latency = 1 };

const std::unordered_map<PerformanceMetric, std::string> PerformanceMetricString{
    {PerformanceMetric::Bandwidth, "bandwidth"},
    {PerformanceMetric::Latency, "latency"},
};

// Bandwidth measurement statistics and result structures
enum class BandwidthStatistics {
    BandwidthMean,
    BandwidthMin,
    BandwidthMax,
    BandwidthStdDev,
    PacketsPerSecondMean,
    CyclesMean
};

// The header of each statistic in the Bandwidth Summary CSV
const std::unordered_map<BandwidthStatistics, std::string> BandwidthStatisticsHeader = {
    {BandwidthStatistics::BandwidthMean, "avg_bandwidth_gigabytes_per_s"},
    {BandwidthStatistics::BandwidthMin, "bw_min_gigabytes_per_s"},
    {BandwidthStatistics::BandwidthMax, "bw_max_gigabytes_per_s"},
    {BandwidthStatistics::BandwidthStdDev, "bw_std_dev_gigabytes_per_s"},
    {BandwidthStatistics::PacketsPerSecondMean, "avg_packets_per_s"},
    {BandwidthStatistics::CyclesMean, "avg_cycles"},
};

// Bandwidth measurement result structures
struct BandwidthResult {
    uint32_t num_devices{};
    uint32_t device_id{};
    RoutingDirection direction = RoutingDirection::NONE;
    uint32_t total_traffic_count{};
    uint32_t num_packets{};
    uint32_t packet_size{};
    uint64_t cycles{};
    double bandwidth_GB_s{};
    double packets_per_second{};
    std::optional<double> telemetry_bw_GB_s_min;
    std::optional<double> telemetry_bw_GB_s_avg;
    std::optional<double> telemetry_bw_GB_s_max;
};

struct BandwidthResultSummary {
    std::string test_name;
    uint32_t num_iterations{};
    std::string ftype;
    std::string ntype;
    std::string topology;
    uint32_t num_links{};
    uint32_t num_packets{};
    std::vector<uint32_t> num_devices;
    uint32_t packet_size{};
    std::vector<double> cycles_vector;
    std::vector<double> bandwidth_vector_GB_s;
    std::vector<double> packets_per_second_vector;
    std::vector<double> statistics_vector;  // Stores the calculated statistics for each test
    uint32_t max_packet_size{};             // Max packet size for router (always set explicitly)

    // Optional fields for database upload CSV
    std::optional<std::string> file_name;
    std::optional<std::string> machine_type;
    std::optional<std::string> test_ts;
};

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
    uint32_t num_iterations{};
    std::string ftype;
    std::string ntype;
    std::string topology;
    uint32_t num_links{};
    uint32_t num_packets{};
    std::vector<uint32_t> num_devices;
    uint32_t packet_size{};
    std::vector<double> cycles_vector;
    std::vector<double> packets_per_second_vector;
    std::vector<double> statistics_vector;  // Stores the calculated statistics for each test
    uint32_t max_packet_size{};             // Max packet size for router (always set explicitly)
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

// Golden CSV comparison structures
struct GoldenCsvEntry {
    std::string test_name;
    std::string ftype;
    std::string ntype;
    std::string topology;
    std::string num_devices;
    uint32_t num_links{};
    uint32_t packet_size{};
    uint32_t num_iterations{};
    double cycles{};
    double bandwidth_GB_s{};
    double packets_per_second{};
    double tolerance_percent{};  // Per-test tolerance percentage
    uint32_t max_packet_size{};  // Max packet size for router (always set from CSV)
};

struct ComparisonResult {
    double speedup() const { return current_bandwidth_GB_s / golden_bandwidth_GB_s; }
    double difference_percent() const {
        return ((current_bandwidth_GB_s - golden_bandwidth_GB_s) / golden_bandwidth_GB_s) * 100.0;
    }

    std::string test_name;
    std::string ftype;
    std::string ntype;
    std::string topology;
    std::string num_devices;
    uint32_t num_links{};
    uint32_t packet_size{};
    uint32_t num_iterations{};
    double current_bandwidth_GB_s{};
    double golden_bandwidth_GB_s{};
    bool within_tolerance{};
    std::string status;
    uint32_t max_packet_size{};  // Max packet size for router (always set from test result)
};

// Used to organize per-test speedups by test topology, packet size, and ntype
struct SpeedupsByTopology {
    Topology topology = Topology::Linear;
    double topology_geomean_speedup{};
    std::unordered_map<uint32_t, std::vector<double>> speedups_by_packet_size;
    std::unordered_map<uint32_t, double> geomean_speedup_by_packet_size;
    std::unordered_map<NocSendType, std::vector<double>> speedups_by_ntype;
    std::unordered_map<NocSendType, double> geomean_speedup_by_ntype;
};

class PostComparisonAnalyzer {
public:
    PostComparisonAnalyzer(const std::vector<ComparisonResult>& comparison_results) :
        comparison_results_(comparison_results) {};

    void generate_comparison_statistics();

    void generate_comparison_statistics_csv(const std::filesystem::path& csv_file_path);

private:
    void organize_speedups_by_topology();

    double calculate_geomean_speedup(const std::vector<double>& speedups);

    void calculate_overall_geomean_speedup();

    std::vector<double> concatenate_topology_speedups(const SpeedupsByTopology& topology_speedups);

    void calculate_geomean_speedup_by_topology();

    const std::vector<ComparisonResult> comparison_results_;
    std::unordered_map<Topology, SpeedupsByTopology> speedups_per_topology_;
    double overall_geomean_speedup_ = 1.0;
};

// Aggregates bandwidth results, generates CSVs, and compares against golden data.
template <typename T, typename U>
class ResultsManager {
protected:
    PerformanceMetric performance_t;
    std::vector<T> results_;
    std::vector<U> results_summary_;
    std::filesystem::path csv_summary_file_path_;
    std::filesystem::path csv_summary_upload_file_path_;

public:
    std::string get_perf_metric_name() const { return PerformanceMetricString.at(performance_t); }
    std::string get_golden_csv_filename();
    void generate_summary_csv();
    void populate_upload_metadata_fields();
    void generate_summary_upload_csv();
    virtual void write_summary_csv_to_file(const std::filesystem::path& csv_path, bool include_upload_columns) = 0;
    virtual ~ResultsManager() = default;
};

class BandwidthResultsManager : public ResultsManager<BandwidthResult, BandwidthResultSummary> {
public:
    BandwidthResultsManager() { performance_t = PerformanceMetric::Bandwidth; };

    void initialize_bandwidth_csv_file(bool telemetry_enabled);
    void add_result(const TestConfig& config, const BandwidthResult& result);
    void add_summary(const TestConfig& config, const BandwidthResultSummary& summary);
    void append_to_csv(const TestConfig& config, const BandwidthResult& result);
    void load_golden_csv();
    void generate_summary();
    void validate_against_golden();
    void setup_ci_artifacts();
    bool has_failures() const;
    std::vector<std::string> get_failed_tests() const;

private:
    // Golden comparison and stats
    std::vector<GoldenCsvEntry> golden_csv_entries_;
    std::vector<ComparisonResult> comparison_results_;
    std::vector<std::string> failed_tests_;
    std::vector<BandwidthStatistics> stat_order_;

    // Paths
    std::filesystem::path csv_file_path_;
    std::filesystem::path diff_csv_file_path_;
    std::filesystem::path comparison_statistics_csv_file_path_;

    bool telemetry_enabled_ = false;
    bool has_failures_ = false;

    // Helpers
    std::string convert_num_devices_to_string(const std::vector<uint32_t>& num_devices) const;
    std::vector<GoldenCsvEntry>::iterator fetch_corresponding_golden_entry(const BandwidthResultSummary& test_result);
    ComparisonResult create_comparison_result(const BandwidthResultSummary& test_result);
    void set_comparison_statistics_csv_file_path();
    void calculate_bandwidth_summary_statistics();
    void calculate_mean(
        const BandwidthStatistics& stat,
        const std::function<const std::vector<double>&(const BandwidthResultSummary&)>& getter);
    void calculate_cycles_mean();
    void calculate_packets_per_second_mean();
    void calculate_bandwidth_mean();
    void calculate_bandwidth_min();
    void calculate_bandwidth_max();
    void calculate_bandwidth_std_dev();
    void write_summary_csv_to_file(const std::filesystem::path& csv_path, bool include_upload_columns) override;
    void populate_comparison_result_bandwidth(
        double result_bandwidth_GB_s,
        ComparisonResult& comp_result,
        std::vector<GoldenCsvEntry>::iterator golden_it,
        const std::vector<GoldenCsvEntry>::iterator& golden_end);
    void populate_comparison_tolerance_and_status(
        ComparisonResult& comp_result,
        std::vector<GoldenCsvEntry>::iterator golden_it,
        const std::vector<GoldenCsvEntry>::iterator& golden_end);
    std::ofstream init_diff_csv_file(
        std::filesystem::path& diff_csv_path, const std::string& csv_header, const std::string& test_type);
    void compare_summary_results_with_golden();

    // Iteration grouping
    std::unordered_map<std::string, size_t> test_name_to_summary_index_;
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

    void initialize_latency_results_csv_file();
    void write_summary_csv_to_file(const std::filesystem::path& csv_path, bool include_upload_columns) override;
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
LatencyResultsManager::LatencyWorkerLocation LatencyResultsManager::find_latency_worker_device(
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
void LatencyResultsManager::populate_comparison_tolerance_and_status(
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

}  // namespace tt::tt_fabric::fabric_tests
