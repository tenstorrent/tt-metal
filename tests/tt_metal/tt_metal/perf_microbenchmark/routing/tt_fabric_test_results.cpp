// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/tt_fabric_test_results.hpp"

namespace tt::tt_fabric::fabric_tests {

void PostComparisonAnalyzer::organize_speedups_by_topology() {
    // Go through each comparison result and add its speedup to the corresponding topology->packet_size and
    // topology->ntype combination
    for (const auto& result : comparison_results_) {
        const std::optional<Topology> result_topology = enchantum::cast<Topology>(result.topology);
        TT_FATAL(result_topology.has_value(), "Invalid topology in comparison result: {}", result.topology);
        const double result_speedup = result.speedup();
        speedups_per_topology_[result_topology.value()].speedups_by_packet_size[result.packet_size].push_back(
            result_speedup);
        std::optional<NocSendType> result_ntype = enchantum::cast<NocSendType>(result.ntype);
        TT_FATAL(result_ntype.has_value(), "Invalid ntype in comparison result: {}", result.ntype);
        speedups_per_topology_[result_topology.value()].speedups_by_ntype[result_ntype.value()].push_back(
            result_speedup);
    }
}

double PostComparisonAnalyzer::calculate_geomean_speedup(const std::vector<double>& speedups) {
    if (speedups.empty()) {
        log_error(tt::LogTest, "No speedups found to calculate geomean speedup, is the speedups vector empty?");
        return 1.0;
    }
    double log_geomean_speedup = 0.0;
    for (const auto& speedup : speedups) {
        log_geomean_speedup += std::log(speedup);
    }
    log_geomean_speedup /= speedups.size();
    double geomean_speedup = std::exp(log_geomean_speedup);
    return geomean_speedup;
}

void PostComparisonAnalyzer::calculate_overall_geomean_speedup() {
    std::vector<double> speedups(comparison_results_.size());
    std::transform(
        comparison_results_.begin(), comparison_results_.end(), speedups.begin(), [](const ComparisonResult& result) {
            return result.speedup();
        });
    overall_geomean_speedup_ = calculate_geomean_speedup(speedups);
    log_info(tt::LogTest, "Overall Geomean Speedup: {:.6f}", overall_geomean_speedup_);
}

std::vector<double> PostComparisonAnalyzer::concatenate_topology_speedups(const SpeedupsByTopology& topology_speedups) {
    // Figure out how many speedups we have
    int num_speedups = 0;
    for (const auto& [packet_size, speedups] : topology_speedups.speedups_by_packet_size) {
        num_speedups += speedups.size();
    }
    if (num_speedups == 0) {
        log_error(tt::LogTest, "No speedups found to concatenate, was topology_speedups correctly populated?");
        return std::vector<double>();
    }
    std::vector<double> concatenated_speedups;
    concatenated_speedups.reserve(num_speedups);
    for (const auto& [packet_size, speedups] : topology_speedups.speedups_by_packet_size) {
        concatenated_speedups.insert(concatenated_speedups.end(), speedups.begin(), speedups.end());
    }
    return concatenated_speedups;
}

void PostComparisonAnalyzer::calculate_geomean_speedup_by_topology() {
    for (auto& [topology, topology_speedups] : speedups_per_topology_) {
        // Calculate geomean speedup by packet size and ntype
        for (const auto& [packet_size, speedups] : topology_speedups.speedups_by_packet_size) {
            topology_speedups.geomean_speedup_by_packet_size[packet_size] = calculate_geomean_speedup(speedups);
            log_info(
                tt::LogTest,
                "Topology: {}, Packet Size: {}, Geomean Speedup: {:.6f}",
                topology,
                packet_size,
                topology_speedups.geomean_speedup_by_packet_size[packet_size]);
        }
        for (const auto& [ntype, speedups] : topology_speedups.speedups_by_ntype) {
            topology_speedups.geomean_speedup_by_ntype[ntype] = calculate_geomean_speedup(speedups);
            log_info(
                tt::LogTest,
                "Topology: {}, NType: {}, Geomean Speedup: {:.6f}",
                topology,
                ntype,
                topology_speedups.geomean_speedup_by_ntype[ntype]);
        }
        // To calculate overall geomean speedup for this topology, we need to merge the speedups of one sub-category
        const std::vector<double> concatenated_speedups = concatenate_topology_speedups(topology_speedups);
        topology_speedups.topology_geomean_speedup = calculate_geomean_speedup(concatenated_speedups);
        log_info(
            tt::LogTest,
            "Topology: {}, Overall Geomean Speedup: {:.6f}",
            topology,
            topology_speedups.topology_geomean_speedup);
    }
}

void PostComparisonAnalyzer::generate_comparison_statistics() {
    // This function calculates overall post-comparison statistics such as geomean speedup, etc.
    // Per-test speedup was calculated in populate_comparison_result_bandwidth()
    // Classify speedup values by topology
    organize_speedups_by_topology();

    // Calculate geometric mean speedup by topology, packet size, and ntype
    calculate_geomean_speedup_by_topology();

    // Calculate geometric mean speedup
    calculate_overall_geomean_speedup();
}

void PostComparisonAnalyzer::generate_comparison_statistics_csv(const std::filesystem::path& csv_file_path) {
    // Create detailed CSV file with header
    std::ofstream comparison_statistics_csv_stream(csv_file_path, std::ios::out | std::ios::trunc);
    if (!comparison_statistics_csv_stream.is_open()) {
        log_error(tt::LogTest, "Failed to create comparison statistics CSV file: {}", csv_file_path.string());
        return;
    }
    // Write detailed header
    comparison_statistics_csv_stream << "topology,packet_size,ntype,geomean_speedup\n";
    log_info(tt::LogTest, "Initialized comparison statistics CSV file: {}", csv_file_path.string());

    // Write most specific speedups first, then overall geomean speedup
    for (const auto& [topology, topology_speedups] : speedups_per_topology_) {
        std::string topology_str = std::string(enchantum::to_string(topology));
        for (const auto& [packet_size, speedup] : topology_speedups.geomean_speedup_by_packet_size) {
            comparison_statistics_csv_stream << topology_str << "," << packet_size << "," << "ALL" << "," << speedup
                                             << "\n";
        }
        for (const auto& [ntype, speedup] : topology_speedups.geomean_speedup_by_ntype) {
            std::string ntype_str = std::string(enchantum::to_string(ntype));
            comparison_statistics_csv_stream << topology_str << "," << "ALL" << "," << ntype_str << "," << speedup
                                             << "\n";
        }
        comparison_statistics_csv_stream << topology_str << "," << "ALL" << "," << "ALL" << ","
                                         << topology_speedups.topology_geomean_speedup << "\n";
    }
    comparison_statistics_csv_stream << "Overall," << "ALL" << "," << "ALL" << "," << overall_geomean_speedup_ << "\n";
    comparison_statistics_csv_stream.close();
    log_info(tt::LogTest, "Comparison statistics CSV results appended to: {}", csv_file_path.string());
}

}  // namespace tt::tt_fabric::fabric_tests
