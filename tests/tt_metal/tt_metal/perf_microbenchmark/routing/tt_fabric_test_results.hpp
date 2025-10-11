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
#include <enchantum/enchantum.hpp>

#include "tt_metal/fabric/fabric_edm_packet_header.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/fabric_edm_types.hpp>
#include <tt-metalium/mesh_graph.hpp>

using Topology = tt::tt_fabric::Topology;
using NocSendType = tt::tt_fabric::NocSendType;
using RoutingDirection = tt::tt_fabric::RoutingDirection;

namespace tt::tt_fabric::fabric_tests {

// Bandwidth measurement result structures
struct BandwidthResult {
    uint32_t num_devices;
    uint32_t device_id;
    RoutingDirection direction;
    uint32_t total_traffic_count;
    uint32_t num_packets;
    uint32_t packet_size;
    uint64_t cycles;
    double bandwidth_GB_s;
    double packets_per_second;
    std::optional<double> telemetry_bw_GB_s_min;
    std::optional<double> telemetry_bw_GB_s_avg;
    std::optional<double> telemetry_bw_GB_s_max;
};

struct BandwidthResultSummary {
    std::string test_name;
    uint32_t num_iterations;
    std::string ftype;
    std::string ntype;
    std::string topology;
    uint32_t num_links;
    uint32_t num_packets;
    std::vector<uint32_t> num_devices;
    uint32_t packet_size;
    std::vector<double> cycles_vector;
    std::vector<double> bandwidth_vector_GB_s;
    std::vector<double> packets_per_second_vector;
    std::vector<double> statistics_vector;  // Stores the calculated statistics for each test
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
};

// Used to organize per-test speedups by test topology, packet size, and ntype
struct SpeedupsByTopology {
    Topology topology;
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

}  // namespace tt::tt_fabric::fabric_tests
