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

namespace tt::tt_fabric::fabric_tests {

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
class BandwidthResultsManager {
public:
    BandwidthResultsManager();

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
    // Accumulators (per-call or aggregated)
    std::vector<BandwidthResult> bandwidth_results_;
    std::vector<BandwidthResultSummary> bandwidth_results_summary_;

    // Golden comparison and stats
    std::vector<GoldenCsvEntry> golden_csv_entries_;
    std::vector<ComparisonResult> comparison_results_;
    std::vector<std::string> failed_tests_;
    std::vector<BandwidthStatistics> stat_order_;

    // Paths
    std::filesystem::path csv_file_path_;
    std::filesystem::path csv_summary_file_path_;
    std::filesystem::path csv_summary_upload_file_path_;
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
    void generate_bandwidth_summary_csv();
    void generate_bandwidth_summary_upload_csv();
    void populate_upload_metadata_fields();
    void write_bandwidth_summary_csv_to_file(const std::filesystem::path& csv_path, bool include_upload_columns);
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
    std::string get_golden_csv_filename();
    void compare_summary_results_with_golden();

    // Iteration grouping
    std::unordered_map<std::string, size_t> test_name_to_summary_index_;
};

}  // namespace tt::tt_fabric::fabric_tests
