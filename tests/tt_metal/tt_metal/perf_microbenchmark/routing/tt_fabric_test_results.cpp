// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <tt-metalium/hal.hpp>
#include <tt-logger/tt-logger.hpp>
#include "impl/context/metal_context.hpp"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/tt_fabric_test_bandwidth_results.hpp"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/tt_fabric_test_config.hpp"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/tt_fabric_test_constants.hpp"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/tt_fabric_test_latency_results.hpp"
#include "tt_fabric_test_results.hpp"

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
    return std::isinf(log_geomean_speedup) ? 1.0 : std::exp(log_geomean_speedup);
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
template <typename T, typename U>
std::string ResultsManager<T, U>::convert_num_devices_to_string(const std::vector<uint32_t>& num_devices) const {
    std::string num_devices_str = "[";
    for (size_t i = 0; i < num_devices.size(); ++i) {
        if (i > 0) {
            num_devices_str += ",";
        }
        num_devices_str += std::to_string(num_devices[i]);
    }
    num_devices_str += "]";
    return num_devices_str;
}

template <typename T, typename U>
void ResultsManager<T, U>::populate_upload_metadata_fields() {
    auto arch_name = tt::tt_metal::hal::get_arch_name();
    auto cluster_type = tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type();
    std::string machine_type = std::string(enchantum::to_string(cluster_type));
    std::transform(machine_type.begin(), machine_type.end(), machine_type.begin(), ::tolower);

    std::string file_name = get_perf_metric_name() + "_" + arch_name + "_" + machine_type;

    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm timestamp_now{};
    localtime_r(&time_t_now, &timestamp_now);
    std::ostringstream timestamp_oss;
    timestamp_oss << std::put_time(&timestamp_now, "%Y-%m-%d %H:%M:%S");
    std::string test_ts = timestamp_oss.str();

    for (auto& result : results_summary_) {
        result.file_name = file_name;
        result.machine_type = machine_type;
        result.test_ts = test_ts;
    }
}

template <typename T, typename U>
void ResultsManager<T, U>::generate_summary_csv() {
    auto arch_name = tt::tt_metal::hal::get_arch_name();
    std::ostringstream summary_oss;
    summary_oss << get_perf_metric_name() + "_summary_results_" << arch_name << ".csv";

    std::filesystem::path output_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        std::string(OUTPUT_DIR);
    csv_summary_file_path_ = output_path / summary_oss.str();

    write_summary_csv_to_file(csv_summary_file_path_, false);
}

template <typename T, typename U>
void ResultsManager<T, U>::generate_summary_upload_csv() {
    auto arch_name = tt::tt_metal::hal::get_arch_name();
    std::ostringstream upload_oss;

    upload_oss << get_perf_metric_name() + "_summary_results_" << arch_name << "_upload.csv";

    std::filesystem::path output_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        std::string(OUTPUT_DIR);
    csv_summary_upload_file_path_ = output_path / upload_oss.str();

    populate_upload_metadata_fields();
    write_summary_csv_to_file(csv_summary_upload_file_path_, true);
}

template <typename T, typename U>
std::ofstream ResultsManager<T, U>::init_diff_csv_file(
    std::filesystem::path& diff_csv_path, const std::string& csv_header, const std::string& test_type) {
    std::filesystem::path output_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        std::string(OUTPUT_DIR);
    std::ostringstream diff_oss;
    auto arch_name = tt::tt_metal::hal::get_arch_name();
    diff_oss << test_type << "_diff_" << arch_name << ".csv";
    diff_csv_path = output_path / diff_oss.str();

    std::ofstream diff_csv_stream(diff_csv_path, std::ios::out | std::ios::trunc);
    if (!diff_csv_stream.is_open()) {
        log_error(tt::LogTest, "Failed to create {} diff CSV file: {}", test_type, diff_csv_path.string());
    } else {
        diff_csv_stream << csv_header << "\n";
        log_info(tt::LogTest, "Initialized {} diff CSV file: {}", test_type, diff_csv_path.string());
    }
    return diff_csv_stream;
}

template <typename T, typename U>
std::string ResultsManager<T, U>::get_golden_csv_filename() {
    auto arch_name = tt::tt_metal::hal::get_arch_name();
    auto cluster_type = tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type();

    // Convert cluster type enum to lowercase string
    std::string cluster_name = std::string(enchantum::to_string(cluster_type));
    std::transform(cluster_name.begin(), cluster_name.end(), cluster_name.begin(), ::tolower);

    std::string file_name = "golden_" + get_perf_metric_name() + "_summary_" + arch_name + "_" + cluster_name + ".csv";
    return file_name;
}

template class ResultsManager<BandwidthResult, BandwidthResultSummary>;
template class ResultsManager<LatencyResult, LatencyResultSummary>;
}  // namespace tt::tt_fabric::fabric_tests
