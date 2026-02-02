// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/tt_fabric_test_results.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <functional>
#include <numeric>
#include <optional>
#include <sstream>
#include <tt-metalium/hal.hpp>
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/tt_fabric_test_common.hpp"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/tt_fabric_test_config.hpp"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/tt_fabric_test_constants.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_fabric_test_results.hpp"

namespace tt::tt_fabric::fabric_tests {

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

void BandwidthResultsManager::initialize_results_csv_file(bool telemetry_enabled) {
    telemetry_enabled_ = telemetry_enabled;

    std::filesystem::path tt_metal_home =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir());
    std::filesystem::path bandwidth_results_path = tt_metal_home / std::string(OUTPUT_DIR);

    if (!std::filesystem::exists(bandwidth_results_path)) {
        std::filesystem::create_directories(bandwidth_results_path);
    }

    // Generate detailed CSV filename
    auto arch_name = tt::tt_metal::hal::get_arch_name();
    std::ostringstream oss;
    oss << get_perf_metric_name() + "_results_" << arch_name << ".csv";
    csv_file_path_ = bandwidth_results_path / oss.str();

    // Create detailed CSV file with header
    std::ofstream csv_stream(csv_file_path_, std::ios::out | std::ios::trunc);  // Truncate file
    if (!csv_stream.is_open()) {
        log_error(tt::LogTest, "Failed to create {} CSV file: {}", get_perf_metric_name(), csv_file_path_.string());
        return;
    }

    // Write detailed header
    csv_stream
        << "test_name,ftype,ntype,topology,num_devices,device,num_links,direction,total_traffic_count,num_packets,"
           "packet_size,cycles,"
           "bandwidth_GB_s,packets_per_second";
    if (telemetry_enabled_) {
        csv_stream << ",telemetry_bw_GB_s_min,telemetry_bw_GB_s_avg,telemetry_bw_GB_s_max";
    }
    csv_stream << "\n";
    csv_stream.close();

    log_info(tt::LogTest, "Initialized {} CSV file: {}", get_perf_metric_name(), csv_file_path_.string());
}

void BandwidthResultsManager::add_result(const TestConfig& config, const BandwidthResult& result) {
    (void)config;
    results_.push_back(result);
}

void BandwidthResultsManager::add_summary(const TestConfig& config, const BandwidthResultSummary& summary) {
    const std::string& test_name = config.name;
    // First iteration or first time we see this test name
    if (config.iteration_number == 0 || !test_name_to_summary_index_.contains(test_name)) {
        test_name_to_summary_index_[test_name] = results_summary_.size();
        results_summary_.push_back(summary);
        return;
    }

    // Append to existing summary
    auto idx = test_name_to_summary_index_.at(test_name);
    BandwidthResultSummary& existing = results_summary_.at(idx);

    existing.num_iterations++;
    existing.cycles_vector.insert(
        existing.cycles_vector.end(), summary.cycles_vector.begin(), summary.cycles_vector.end());
    existing.bandwidth_vector_GB_s.insert(
        existing.bandwidth_vector_GB_s.end(),
        summary.bandwidth_vector_GB_s.begin(),
        summary.bandwidth_vector_GB_s.end());
    existing.packets_per_second_vector.insert(
        existing.packets_per_second_vector.end(),
        summary.packets_per_second_vector.begin(),
        summary.packets_per_second_vector.end());
}

void BandwidthResultsManager::append_to_csv(const TestConfig& config, const BandwidthResult& result) {
    // Extract representative ftype and ntype from first sender's first pattern
    const TrafficPatternConfig& first_pattern = fetch_first_traffic_pattern(config);
    std::string ftype_str = fetch_pattern_ftype(first_pattern);
    std::string ntype_str = fetch_pattern_ntype(first_pattern);

    // Open CSV file in append mode
    std::ofstream csv_stream(csv_file_path_, std::ios::out | std::ios::app);
    if (!csv_stream.is_open()) {
        log_error(tt::LogTest, "Failed to open CSV file for appending: {}", csv_file_path_.string());
        return;
    }

    csv_stream << config.parametrized_name << "," << ftype_str << "," << ntype_str << ","
               << enchantum::to_string(config.fabric_setup.topology) << "," << result.num_devices << ","
               << result.device_id << "," << config.fabric_setup.num_links << ","
               << enchantum::to_string(result.direction) << "," << result.total_traffic_count << ","
               << result.num_packets << "," << result.packet_size << "," << result.cycles << "," << std::fixed
               << std::setprecision(6) << result.bandwidth_GB_s << "," << std::fixed << std::setprecision(3)
               << result.packets_per_second;

    if (telemetry_enabled_ && result.telemetry_bw_GB_s_min.has_value()) {
        csv_stream << "," << std::fixed << std::setprecision(3) << result.telemetry_bw_GB_s_min.value() << ","
                   << std::fixed << std::setprecision(3) << result.telemetry_bw_GB_s_avg.value() << "," << std::fixed
                   << std::setprecision(3) << result.telemetry_bw_GB_s_max.value();
    }
    csv_stream << "\n";

    csv_stream.close();
    log_info(tt::LogTest, "Bandwidth results appended to CSV file: {}", csv_file_path_.string());
}

void BandwidthResultsManager::load_golden_csv() {
    golden_csv_entries_.clear();

    std::string golden_filename = get_golden_csv_filename();
    std::filesystem::path golden_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_metal/perf_microbenchmark/routing/golden" / golden_filename;

    if (!std::filesystem::exists(golden_path)) {
        log_warning(tt::LogTest, "Golden CSV file not found: {}", golden_path.string());
        return;
    }

    std::ifstream golden_file(golden_path);
    if (!golden_file.is_open()) {
        log_error(tt::LogTest, "Failed to open golden CSV file: {}", golden_path.string());
        return;
    }

    std::string line;
    bool is_header = true;
    while (std::getline(golden_file, line)) {
        if (is_header) {
            is_header = false;
            continue;  // Skip header
        }

        std::istringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;

        // Parse CSV line
        while (std::getline(ss, token, ',')) {
            // Handle quoted strings for num_devices
            if (!token.empty() && token.front() == '"' && token.back() != '"') {
                std::string quoted_token = token;
                while (std::getline(ss, token, ',') && token.back() != '"') {
                    quoted_token += "," + token;
                }
                quoted_token += "," + token;
                // Remove quotes
                quoted_token = quoted_token.substr(1, quoted_token.length() - 2);
                tokens.push_back(quoted_token);
            } else if (!token.empty() && token.front() == '"' && token.back() == '"') {
                // Remove quotes from single quoted token
                tokens.push_back(token.substr(1, token.length() - 2));
            } else {
                tokens.push_back(token);
            }
        }

        // Validate we have enough tokens for the new format with tolerance and max_packet_size
        if (tokens.size() < 16) {
            log_error(tt::LogTest, "Invalid CSV format in golden file. Expected 16 fields, got {}", tokens.size());
            continue;
        }

        GoldenCsvEntry entry;
        entry.test_name = tokens[0];
        entry.ftype = tokens[1];
        entry.ntype = tokens[2];
        entry.topology = tokens[3];
        entry.num_devices = tokens[4];
        entry.num_links = std::stoul(tokens[5]);
        entry.packet_size = std::stoul(tokens[6]);
        entry.num_iterations = std::stoul(tokens[7]);
        entry.cycles = std::stod(tokens[8]);
        entry.packets_per_second = std::stod(tokens[9]);
        entry.bandwidth_GB_s = std::stod(tokens[10]);
        // Skip min, max, std dev (indexes 11,12,13)
        entry.tolerance_percent = std::stod(tokens[14]);
        entry.max_packet_size = std::stoul(tokens[15]);
        golden_csv_entries_.push_back(entry);
    }

    golden_file.close();
    log_info(tt::LogTest, "Loaded {} golden entries from: {}", golden_csv_entries_.size(), golden_path.string());
}

void BandwidthResultsManager::generate_summary() {
    calculate_bandwidth_summary_statistics();
    generate_summary_csv();
    generate_summary_upload_csv();
    validate_against_golden();
}

void BandwidthResultsManager::validate_against_golden() {
    if (golden_csv_entries_.empty()) {
        log_warning(tt::LogTest, "Skipping golden CSV comparison - no golden file found");
        has_failures_ = false;
        return;
    }

    comparison_results_.clear();
    failed_tests_.clear();
    compare_summary_results_with_golden();
    has_failures_ = !failed_tests_.empty();
}

void BandwidthResultsManager::setup_ci_artifacts() {
    std::filesystem::path tt_metal_home =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir());
    std::filesystem::path ci_artifacts_path = tt_metal_home / std::string(CI_ARTIFACTS_DIR);

    if (!std::filesystem::exists(ci_artifacts_path)) {
        std::filesystem::create_directories(ci_artifacts_path);
    }

    std::vector<std::filesystem::path> csv_paths = {
        csv_file_path_, csv_summary_file_path_, csv_summary_upload_file_path_, diff_csv_file_path_};
    for (const auto& csv_filepath : csv_paths) {
        if (csv_filepath.empty() || !std::filesystem::exists(csv_filepath)) {
            continue;
        }
        try {
            std::filesystem::copy_file(
                csv_filepath,
                ci_artifacts_path / csv_filepath.filename(),
                std::filesystem::copy_options::overwrite_existing);
        } catch (const std::filesystem::filesystem_error& e) {
            log_debug(
                tt::LogTest,
                "Failed to copy CSV file {} to CI artifacts directory: {}",
                csv_filepath.filename().string(),
                e.what());
        }
    }
    log_trace(tt::LogTest, "Copied CSV files to CI artifacts directory: {}", ci_artifacts_path.string());
}

bool BandwidthResultsManager::has_failures() const { return has_failures_; }

std::vector<std::string> BandwidthResultsManager::get_failed_tests() const { return failed_tests_; }

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

std::vector<GoldenCsvEntry>::iterator BandwidthResultsManager::fetch_corresponding_golden_entry(
    const BandwidthResultSummary& test_result) {
    std::string num_devices_str = convert_num_devices_to_string(test_result.num_devices);
    auto golden_it =
        std::find_if(golden_csv_entries_.begin(), golden_csv_entries_.end(), [&](const GoldenCsvEntry& golden) {
            return golden.test_name == test_result.test_name && golden.ftype == test_result.ftype &&
                   golden.ntype == test_result.ntype && golden.topology == test_result.topology &&
                   golden.num_devices == num_devices_str && golden.num_links == test_result.num_links &&
                   golden.packet_size == test_result.packet_size;
        });
    return golden_it;
}

ComparisonResult BandwidthResultsManager::create_comparison_result(const BandwidthResultSummary& test_result) {
    std::string num_devices_str = convert_num_devices_to_string(test_result.num_devices);
    ComparisonResult comp_result;
    comp_result.test_name = test_result.test_name;
    comp_result.ftype = test_result.ftype;
    comp_result.ntype = test_result.ntype;
    comp_result.topology = test_result.topology;
    comp_result.num_devices = num_devices_str;
    comp_result.num_links = test_result.num_links;
    comp_result.packet_size = test_result.packet_size;
    comp_result.num_iterations = test_result.num_iterations;
    comp_result.max_packet_size = test_result.max_packet_size;
    return comp_result;
}

void BandwidthResultsManager::set_comparison_statistics_csv_file_path() {
    std::ostringstream comparison_statistics_oss;
    auto arch_name = tt::tt_metal::hal::get_arch_name();
    comparison_statistics_oss << "bandwidth_comparison_statistics_" << arch_name << ".csv";
    std::filesystem::path output_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        std::string(OUTPUT_DIR);
    comparison_statistics_csv_file_path_ = output_path / comparison_statistics_oss.str();
}

void BandwidthResultsManager::calculate_mean(
    const BandwidthStatistics& stat,
    const std::function<const std::vector<double>&(const BandwidthResultSummary&)>& getter) {
    stat_order_.push_back(stat);
    for (auto& result : results_summary_) {
        const std::vector<double>& measurements_vector = getter(result);
        double sum = std::accumulate(measurements_vector.begin(), measurements_vector.end(), 0.0);
        double mean = sum / result.num_iterations;
        result.statistics_vector.push_back(mean);
    }
}

void BandwidthResultsManager::calculate_cycles_mean() {
    calculate_mean(
        BandwidthStatistics::CyclesMean, [](const auto& result) -> const auto& { return result.cycles_vector; });
}

void BandwidthResultsManager::calculate_packets_per_second_mean() {
    calculate_mean(BandwidthStatistics::PacketsPerSecondMean, [](const auto& result) -> const auto& {
        return result.packets_per_second_vector;
    });
}

void BandwidthResultsManager::calculate_bandwidth_mean() {
    calculate_mean(BandwidthStatistics::BandwidthMean, [](const auto& result) -> const auto& {
        return result.bandwidth_vector_GB_s;
    });
}

void BandwidthResultsManager::calculate_bandwidth_min() {
    stat_order_.push_back(BandwidthStatistics::BandwidthMin);
    for (auto& result : results_summary_) {
        result.statistics_vector.push_back(
            *std::min_element(result.bandwidth_vector_GB_s.begin(), result.bandwidth_vector_GB_s.end()));
    }
}

void BandwidthResultsManager::calculate_bandwidth_max() {
    stat_order_.push_back(BandwidthStatistics::BandwidthMax);
    for (auto& result : results_summary_) {
        result.statistics_vector.push_back(
            *std::max_element(result.bandwidth_vector_GB_s.begin(), result.bandwidth_vector_GB_s.end()));
    }
}

void BandwidthResultsManager::calculate_bandwidth_std_dev() {
    stat_order_.push_back(BandwidthStatistics::BandwidthStdDev);
    for (auto& result : results_summary_) {
        double sum = std::accumulate(result.bandwidth_vector_GB_s.begin(), result.bandwidth_vector_GB_s.end(), 0.0);
        double mean = sum / result.num_iterations;
        double variance = 0.0;
        for (auto& bandwidth_gb_s : result.bandwidth_vector_GB_s) {
            variance += std::pow(bandwidth_gb_s - mean, 2);
        }
        variance /= result.num_iterations;
        double std_dev = std::sqrt(variance);
        result.statistics_vector.push_back(std_dev);
    }
}

void BandwidthResultsManager::calculate_bandwidth_summary_statistics() {
    // Add new statistics here
    // The statistics will be displayed in the bandwidth summary CSV file in this order
    // The name of each statistic collected is maintained in-order in the stat_order_ vector
    // The statistics are calculated for each test in the same order and are stored in each test's
    // BandwidthResultSummary.statistics_vector Each function here should calculate the statistics for every test
    // within a single invocation (see functions for details) NOTE: If you add new statistics, you must re-generate
    // the golden CSV file, otherwise benchmarking will fail.
    calculate_cycles_mean();
    calculate_packets_per_second_mean();
    calculate_bandwidth_mean();
    calculate_bandwidth_min();
    calculate_bandwidth_max();
    calculate_bandwidth_std_dev();
}

void BandwidthResultsManager::write_summary_csv_to_file(
    const std::filesystem::path& csv_path, bool include_upload_columns) {
    std::ofstream csv_stream(csv_path, std::ios::out | std::ios::trunc);
    if (!csv_stream.is_open()) {
        log_error(tt::LogTest, "Failed to create CSV file: {}", csv_path.string());
        return;
    }

    if (include_upload_columns) {
        csv_stream << "file_name,machine_type,test_ts,";
    }
    csv_stream << "test_name,ftype,ntype,topology,num_devices,num_links,packet_size,iterations";
    for (BandwidthStatistics stat : stat_order_) {
        const std::string& stat_name = BandwidthStatisticsHeader.at(stat);
        csv_stream << "," << stat_name;
    }
    csv_stream << ",tolerance_percent\n";
    log_info(tt::LogTest, "Initialized CSV file: {}", csv_path.string());

    for (const auto& result : results_summary_) {
        if (include_upload_columns) {
            csv_stream << result.file_name.value() << "," << result.machine_type.value() << ","
                       << result.test_ts.value() << ",";
        }

        std::string num_devices_str = convert_num_devices_to_string(result.num_devices);

        csv_stream << result.test_name << "," << result.ftype << "," << result.ntype << "," << result.topology << ",\""
                   << num_devices_str << "\"," << result.num_links << "," << result.packet_size << ","
                   << result.num_iterations;
        for (double stat : result.statistics_vector) {
            csv_stream << "," << std::fixed << std::setprecision(6) << stat;
        }

        auto golden_it = fetch_corresponding_golden_entry(result);
        if (golden_it == golden_csv_entries_.end()) {
            csv_stream << "," << 1.0;
        } else {
            csv_stream << "," << golden_it->tolerance_percent;
        }
        csv_stream << "\n";
    }
    csv_stream.close();
    log_info(tt::LogTest, "Bandwidth summary results written to CSV file: {}", csv_path.string());
}

template <typename T, typename U>
void ResultsManager<T, U>::populate_upload_metadata_fields() {
    auto arch_name = tt::tt_metal::hal::get_arch_name();
    auto cluster_type = tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type();
    std::string machine_type = std::string(enchantum::to_string(cluster_type));
    std::transform(machine_type.begin(), machine_type.end(), machine_type.begin(), ::tolower);

    // !<<<<<<<<<<<<<<<<<<<<<<<<<<< Remember that this affects schema on supserset side make sure to
    // change>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>!
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

void BandwidthResultsManager::populate_comparison_result_bandwidth(
    double result_bandwidth_GB_s,
    ComparisonResult& comp_result,
    std::vector<GoldenCsvEntry>::iterator golden_it,
    const std::vector<GoldenCsvEntry>::iterator& golden_end) {
    comp_result.current_bandwidth_GB_s = result_bandwidth_GB_s;

    if (golden_it != golden_end) {
        comp_result.golden_bandwidth_GB_s = golden_it->bandwidth_GB_s;
    } else {
        comp_result.golden_bandwidth_GB_s = 0.0;
    }

    populate_comparison_tolerance_and_status(comp_result, golden_it, golden_end);
}

void BandwidthResultsManager::populate_comparison_tolerance_and_status(
    ComparisonResult& comp_result,
    std::vector<GoldenCsvEntry>::iterator golden_it,
    const std::vector<GoldenCsvEntry>::iterator& golden_end) {
    double test_tolerance = 1.0;
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

void BandwidthResultsManager::compare_summary_results_with_golden() {
    if (golden_csv_entries_.empty()) {
        log_warning(tt::LogTest, "Skipping golden CSV comparison - no golden file found");
        return;
    }
    if (results_summary_.size() != golden_csv_entries_.size()) {
        log_warning(
            tt::LogTest,
            "Number of test results ({}) does not match number of golden entries ({})",
            results_summary_.size(),
            golden_csv_entries_.size());
    }

    for (auto& test_result : results_summary_) {
        auto bandwidth_stat_location =
            std::find(stat_order_.begin(), stat_order_.end(), BandwidthStatistics::BandwidthMean);
        if (bandwidth_stat_location == stat_order_.end()) {
            log_error(tt::LogTest, "Average bandwidth statistic not found, was it calculated?");
            return;
        }

        size_t bandwidth_stat_index = std::distance(stat_order_.begin(), bandwidth_stat_location);
        double average_bandwidth_GB_s = test_result.statistics_vector[bandwidth_stat_index];

        ComparisonResult comp_result = create_comparison_result(test_result);
        auto golden_it = fetch_corresponding_golden_entry(test_result);
        populate_comparison_result_bandwidth(average_bandwidth_GB_s, comp_result, golden_it, golden_csv_entries_.end());

        comparison_results_.push_back(comp_result);

        if (!comp_result.within_tolerance && comp_result.status != "NO_GOLDEN") {
            std::ostringstream oss;
            oss << comp_result.test_name << " [" << comp_result.status << "]";
            failed_tests_.push_back(oss.str());
        }
    }

    auto diff_csv_stream = init_diff_csv_file(
        diff_csv_file_path_,
        "test_name,ftype,ntype,topology,num_devices,num_links,packet_size,iterations,"
        "current_bandwidth_GB_s,golden_bandwidth_GB_s,difference_percent,status,max_packet_size",
        "bandwidth");

    if (diff_csv_stream.is_open()) {
        for (const auto& result : comparison_results_) {
            diff_csv_stream << result.test_name << "," << result.ftype << "," << result.ntype << "," << result.topology
                            << ",\"" << result.num_devices << "\"," << result.num_links << "," << result.packet_size
                            << "," << result.num_iterations << "," << std::fixed << std::setprecision(6)
                            << result.current_bandwidth_GB_s << "," << result.golden_bandwidth_GB_s << ","
                            << std::setprecision(2) << result.difference_percent() << "," << result.status << ","
                            << result.max_packet_size << "\n";
        }
        diff_csv_stream.close();
        log_info(tt::LogTest, "Bandwidth comparison diff CSV results written to: {}", diff_csv_file_path_.string());
    }

    PostComparisonAnalyzer post_comparison_analyzer(comparison_results_);
    post_comparison_analyzer.generate_comparison_statistics();
    set_comparison_statistics_csv_file_path();
    post_comparison_analyzer.generate_comparison_statistics_csv(comparison_statistics_csv_file_path_);
}

// ==============================================LATENCY TEST
// MANAGER======================================================

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <tt-logger/tt-logger.hpp>
#include "impl/context/metal_context.hpp"

LatencyResultsManager::LatencyResultsManager(TestFixture& fixture, SenderMemoryMap& sender_memory_map) :
    fixture_(fixture), sender_memory_map_(sender_memory_map) {
    performance_t = PerformanceMetric::Latency;
}

void LatencyResultsManager::create_latency_kernels_for_device(
    TestDevice& test_device, std::unordered_map<MeshCoordinate, TestDevice>& test_devices) {
    const auto& senders = test_device.get_senders();
    const auto& receivers = test_device.get_receivers();

    bool has_sender = !senders.empty();
    bool has_receiver = !receivers.empty();

    if (has_sender) {
        TT_FATAL(senders.size() == 1, "Latency test should have exactly one sender per device");
        const auto& [sender_core, sender_worker] = *senders.begin();
        const auto& sender_configs = sender_worker.get_configs();
        TT_FATAL(!sender_configs.empty(), "Latency sender should have at least one config");

        const auto& sender_config = sender_configs[0].first;
        FabricNodeId responder_device_id = sender_config.dst_node_ids[0];

        TestDevice* responder_device = nullptr;
        for (auto& [responder_coord, responder_test_device] : test_devices) {
            if (responder_test_device.get_node_id() == responder_device_id) {
                responder_device = &responder_test_device;
                (void)responder_coord;
                break;
            }
        }
        TT_FATAL(
            responder_device != nullptr,
            "Could not find responder device with node_id {}",
            responder_device_id.chip_id);

        CoreCoord responder_virtual_core =
            responder_device->get_device_info_provider()->get_virtual_core_from_logical_core(
                sender_config.dst_logical_core);

        test_device.create_latency_sender_kernel(
            sender_core,
            responder_device_id,
            sender_config.parameters.payload_size_bytes,
            sender_config.parameters.num_packets,
            sender_config.parameters.noc_send_type,
            responder_virtual_core);
    } else if (has_receiver) {
        TT_FATAL(receivers.size() == 1, "Latency test should have exactly one receiver per device");
        const auto& [receiver_core, receiver_worker] = *receivers.begin();

        auto sender_location = get_latency_sender_location(test_devices);
        TestDevice* sender_device = sender_location.device;
        CoreCoord sender_core = sender_location.core;

        const auto& sender_senders = sender_device->get_senders();
        const auto& sender_worker = sender_senders.at(sender_core);
        const auto& sender_configs = sender_worker.get_configs();
        const auto& sender_config = sender_configs[0].first;

        uint32_t payload_size = sender_config.parameters.payload_size_bytes;
        uint32_t num_samples = sender_config.parameters.num_packets;
        NocSendType noc_send_type = sender_config.parameters.noc_send_type;
        FabricNodeId sender_device_id = sender_config.src_node_id;

        uint32_t sender_send_buffer_address = sender_device->get_latency_send_buffer_address();
        uint32_t sender_receive_buffer_address = sender_device->get_latency_receive_buffer_address(payload_size);

        CoreCoord sender_virtual_core =
            sender_device->get_device_info_provider()->get_virtual_core_from_logical_core(sender_core);

        test_device.create_latency_responder_kernel(
            receiver_core,
            sender_device_id,
            payload_size,
            num_samples,
            noc_send_type,
            sender_send_buffer_address,
            sender_receive_buffer_address,
            sender_virtual_core);
    } else {
        test_device.create_kernels();
    }
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

void LatencyResultsManager::setup_latency_test_mode(const TestConfig& config) {
    TT_FATAL(
        config.performance_test_mode == PerformanceTestMode::LATENCY,
        "setup_latency_test_mode called when latency test mode is not enabled");

    // Validate that latency tests don't use multiple iterations unless from a sequential pattern
    if (!config.from_sequential_pattern) {
        TT_FATAL(
            config.iteration_number == 1 || config.iteration_number == 0,
            "Latency tests do not support multiple iterations. Use num_packets in the test config instead to "
            "collect multiple samples. Got {} iterations.",
            config.iteration_number);
    }

    // Validate latency test structure
    TT_FATAL(config.senders.size() == 1, "Latency test mode requires exactly one sender");
    TT_FATAL(config.senders[0].patterns.size() == 1, "Latency test mode requires exactly one pattern");

    const auto& sender = config.senders[0];
    const auto& pattern = sender.patterns[0];
    const auto& dest = pattern.destination.value();

    log_info(
        tt::LogTest,
        "Latency test mode: sender={}, responder={}, payload={} bytes, samples={}",
        sender.device.chip_id,
        dest.device.value().chip_id,
        pattern.size.value(),
        pattern.num_packets.value());
}

void LatencyResultsManager::setup_latency_test_workers(
    TestConfig& config, std::unordered_map<MeshCoordinate, TestDevice>& test_devices) {
    log_debug(tt::LogTest, "Latency test mode: manually populating sender and receiver workers");

    // Latency tests have exactly one sender with one pattern
    TT_FATAL(config.senders.size() == 1, "Latency test mode requires exactly one sender");
    TT_FATAL(config.senders[0].patterns.size() == 1, "Latency test mode requires exactly one pattern");

    const auto& sender = config.senders[0];
    const auto& pattern = sender.patterns[0];
    const auto& dest = pattern.destination.value();

    // Use default core if not specified (latency tests typically use a fixed core)
    // TODO: Implement core sweep for sequential_neighbor_exchange
    CoreCoord sender_core = sender.core.value_or(CoreCoord{0, 0});
    CoreCoord receiver_core = dest.core.value_or(CoreCoord{0, 0});
    FabricNodeId sender_device_id = sender.device;
    FabricNodeId receiver_device_id = dest.device.value();

    // Create sender worker on sender device
    if (fixture_.is_local_fabric_node_id(sender_device_id)) {
        const auto& sender_coord = fixture_.get_device_coord(sender_device_id);
        auto& sender_test_device = test_devices.at(sender_coord);

        // Create latency sender config with actual parameters
        TrafficParameters latency_traffic_params = {
            .chip_send_type = pattern.ftype.value(),
            .noc_send_type = pattern.ntype.value(),
            .payload_size_bytes = pattern.size.value(),
            .num_packets = pattern.num_packets.value(),
            .atomic_inc_val = pattern.atomic_inc_val,
            .mcast_start_hops = std::nullopt,
            .enable_flow_control = false,
            .seed = config.seed,
            .is_2D_routing_enabled = fixture_.is_2D_routing_enabled(),
            .mesh_shape = fixture_.get_mesh_shape(),
            .topology = fixture_.get_topology()};

        TestTrafficSenderConfig latency_sender_config = {
            .parameters = latency_traffic_params,
            .src_node_id = sender_device_id,
            .dst_node_ids = {receiver_device_id},
            .hops = std::nullopt,
            .mcast_start_node_id = std::nullopt,
            .dst_logical_core = receiver_core,
            .target_address = 0,
            .atomic_inc_address = std::nullopt,
            .dst_noc_encoding = fixture_.get_worker_noc_encoding(receiver_core),
            .payload_buffer_size = 0,
            .link_id = 0};

        sender_test_device.add_sender_traffic_config(sender_core, std::move(latency_sender_config));

        // Set latency sender kernel
        sender_test_device.set_sender_kernel_src(
            sender_core, "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_latency_sender.cpp");

        log_debug(
            tt::LogTest, "Created latency sender worker on device {} core {}", sender_device_id.chip_id, sender_core);
    }

    // Create receiver worker on receiver device
    if (fixture_.is_local_fabric_node_id(receiver_device_id)) {
        const auto& receiver_coord = fixture_.get_device_coord(receiver_device_id);
        auto& receiver_test_device = test_devices.at(receiver_coord);

        // Create dummy receiver config just to populate the worker
        TestTrafficReceiverConfig dummy_receiver_config = {
            .parameters = TrafficParameters{},
            .sender_id = 0,
            .target_address = 0,
            .atomic_inc_address = std::nullopt,
            .payload_buffer_size = 0,
            .link_id = 0};

        receiver_test_device.add_receiver_traffic_config(receiver_core, dummy_receiver_config);

        // Set latency responder kernel
        receiver_test_device.set_receiver_kernel_src(
            receiver_core,
            "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_latency_responder.cpp");

        log_debug(
            tt::LogTest,
            "Created latency receiver worker on device {} core {}",
            receiver_device_id.chip_id,
            receiver_core);
    }
}

LatencyResultsManager::LatencyWorkerLocation LatencyResultsManager::get_latency_sender_location(
    std::unordered_map<MeshCoordinate, TestDevice>& test_devices) {
    return find_latency_worker_device(
        test_devices, [](TestDevice& d) -> const auto& { return d.get_senders(); }, "sender");
}

LatencyResultsManager::LatencyWorkerLocation LatencyResultsManager::get_latency_receiver_location(
    std::unordered_map<MeshCoordinate, TestDevice>& test_devices) {
    return find_latency_worker_device(
        test_devices, [](TestDevice& d) -> const auto& { return d.get_receivers(); }, "receiver");
}

void LatencyResultsManager::collect_latency_results(std::unordered_map<MeshCoordinate, TestDevice>& test_devices) {
    log_info(tt::LogTest, "Collecting latency results from sender and responder devices");

    // Find sender and responder locations
    auto sender_location = get_latency_sender_location(test_devices);
    auto responder_location = get_latency_receiver_location(test_devices);

    // Get num_samples from sender config
    const auto& sender_configs = sender_location.device->get_senders().begin()->second.get_configs();
    uint32_t num_samples = sender_configs[0].first.parameters.num_packets;
    uint32_t result_buffer_size = num_samples * sizeof(uint32_t);

    // Read latency samples from sender device
    fixture_.read_buffer_from_cores(
        sender_location.mesh_coord,
        {sender_location.core},
        sender_memory_map_.get_result_buffer_address(),
        result_buffer_size);

    // Read responder timestamps from responder device
    fixture_.read_buffer_from_cores(
        responder_location.mesh_coord,
        {responder_location.core},
        sender_memory_map_.get_result_buffer_address(),
        result_buffer_size);

    log_info(tt::LogTest, "Collected {} latency samples from sender and responder", num_samples);
}

void LatencyResultsManager::report_latency_results(
    const TestConfig& config, std::unordered_map<MeshCoordinate, TestDevice>& test_devices) {
    log_info(tt::LogTest, "Reporting latency results for test: {}", config.parametrized_name);

    // Find sender and responder locations
    auto sender_location = get_latency_sender_location(test_devices);
    auto responder_location = get_latency_receiver_location(test_devices);

    const TestDevice* sender_device = sender_location.device;
    MeshCoordinate sender_coord = sender_location.mesh_coord;
    CoreCoord sender_core = sender_location.core;
    FabricNodeId sender_node_id = sender_location.node_id;

    MeshCoordinate responder_coord = responder_location.mesh_coord;
    CoreCoord responder_core = responder_location.core;
    FabricNodeId responder_node_id = responder_location.node_id;

    // Get latency parameters from sender config
    const auto& sender_configs = sender_device->get_senders().begin()->second.get_configs();
    const auto& sender_config = sender_configs[0].first;

    uint32_t num_samples = sender_config.parameters.num_packets;
    uint32_t payload_size = sender_config.parameters.payload_size_bytes;

    // Calculate number of hops between sender and responder
    uint32_t num_hops_to_responder = 0;
    auto hops_map = fixture_.get_hops_to_chip(sender_node_id, responder_node_id);
    for (const auto& [dir, hop_count] : hops_map) {
        (void)dir;
        num_hops_to_responder += hop_count;
    }
    TT_FATAL(num_hops_to_responder != 0, "Number of hops to responder is 0");
    // Multiply by 2 for round-trip (sender -> responder -> sender)
    num_hops_to_responder *= 2;

    uint32_t result_buffer_size = num_samples * sizeof(uint32_t);

    // Read latency samples from sender device
    auto sender_result_data = fixture_.read_buffer_from_cores(
        sender_coord, {sender_core}, sender_memory_map_.get_result_buffer_address(), result_buffer_size);
    const auto& sender_data = sender_result_data.at(sender_core);

    // Read responder elapsed times from responder device
    auto responder_result_data = fixture_.read_buffer_from_cores(
        responder_coord, {responder_core}, sender_memory_map_.get_result_buffer_address(), result_buffer_size);
    const auto& responder_data = responder_result_data.at(responder_core);

    // Parse elapsed times and compute latencies
    // Data is stored as uint32_t elapsed times (in cycles)
    std::vector<uint64_t> raw_latencies_cycles;
    std::vector<uint64_t> responder_times_cycles;
    std::vector<uint64_t> net_latencies_cycles;
    std::vector<uint64_t> per_hop_latency_cycles;

    raw_latencies_cycles.reserve(num_samples);
    responder_times_cycles.reserve(num_samples);
    net_latencies_cycles.reserve(num_samples);
    per_hop_latency_cycles.reserve(num_samples);

    for (uint32_t i = 0; i < num_samples; i++) {
        // Read elapsed times directly (already computed on device)
        uint64_t raw_latency = sender_data[i];
        uint64_t responder_time = responder_data[i];

        // Validate that responder time is reasonable
        TT_FATAL(raw_latency > 0, "Invalid sender latency (zero) for sample {}", i);
        TT_FATAL(responder_time > 0, "Invalid responder time (zero) for sample {}", i);

        // Check for clock synchronization issues between sender and responder devices
        // If responder time exceeds raw latency, this indicates unsynchronized clocks
        if (responder_time >= raw_latency) {
            log_warning(
                tt::LogTest,
                "Sample {}: Responder time ({} cycles) exceeds raw latency ({} cycles). "
                "This indicates clock drift/skew between devices. Sender and responder timestamps "
                "cannot be directly compared without clock synchronization.",
                i,
                responder_time,
                raw_latency);
            TT_FATAL(
                false,
                "Clock synchronization issue detected: responder processing time cannot exceed round-trip time. "
                "The sender device clock and responder device clock are not synchronized.");
        }

        uint64_t net_latency = raw_latency - responder_time;
        uint64_t per_hop_latency = net_latency / num_hops_to_responder;

        raw_latencies_cycles.push_back(raw_latency);
        responder_times_cycles.push_back(responder_time);
        net_latencies_cycles.push_back(net_latency);
        per_hop_latency_cycles.push_back(per_hop_latency);
    }

    if (raw_latencies_cycles.empty()) {
        log_warning(tt::LogTest, "No valid latency samples collected");
        return;
    }

    // Sort for percentile calculation
    std::sort(raw_latencies_cycles.begin(), raw_latencies_cycles.end());
    std::sort(responder_times_cycles.begin(), responder_times_cycles.end());
    std::sort(net_latencies_cycles.begin(), net_latencies_cycles.end());
    std::sort(per_hop_latency_cycles.begin(), per_hop_latency_cycles.end());

    // Get device frequency for conversion to ns
    uint32_t freq_mhz = fixture_.get_device_frequency_mhz(sender_node_id);
    double freq_ghz = static_cast<double>(freq_mhz) / 1000.0;
    double ns_per_cycle = 1.0 / freq_ghz;

    // Helper lambda to calculate statistics
    auto calc_stats = [](const std::vector<uint64_t>& data) {
        struct Stats {
            uint64_t min, max, p50, p99;
            double avg;
        };
        Stats stats{};
        stats.min = data.front();
        stats.max = data.back();
        uint64_t sum = std::accumulate(data.begin(), data.end(), 0ULL);
        stats.avg = static_cast<double>(sum) / data.size();
        stats.p50 = data[data.size() / 2];
        stats.p99 = data[static_cast<size_t>(data.size() * 0.99)];
        return stats;
    };

    auto raw_stats = calc_stats(raw_latencies_cycles);
    auto responder_stats = calc_stats(responder_times_cycles);
    auto net_stats = calc_stats(net_latencies_cycles);
    auto per_hop_latency_stats = calc_stats(per_hop_latency_cycles);

    // Log results in table format
    log_info(tt::LogTest, "");
    log_info(tt::LogTest, "=== Latency Test Results for {} ===", config.parametrized_name);
    log_info(tt::LogTest, "Payload size: {} bytes | Num samples: {}", payload_size, raw_latencies_cycles.size());
    log_info(tt::LogTest, "");
    log_info(
        tt::LogTest, "Metric    Raw Latency (ns)    Responder Time (ns)    Net Latency (ns)    Per-Hop Latency (ns)");
    log_info(tt::LogTest, "------------------------------------------------------------------------");
    log_info(
        tt::LogTest,
        "Min       {:>15.2f}    {:>18.2f}    {:>15.2f}    {:>15.2f}",
        raw_stats.min * ns_per_cycle,
        responder_stats.min * ns_per_cycle,
        net_stats.min * ns_per_cycle,
        per_hop_latency_stats.min * ns_per_cycle);
    log_info(
        tt::LogTest,
        "Max       {:>15.2f}    {:>18.2f}    {:>15.2f}    {:>15.2f}",
        raw_stats.max * ns_per_cycle,
        responder_stats.max * ns_per_cycle,
        net_stats.max * ns_per_cycle,
        per_hop_latency_stats.max * ns_per_cycle);
    log_info(
        tt::LogTest,
        "Avg       {:>15.2f}    {:>18.2f}    {:>15.2f}    {:>15.2f}",
        raw_stats.avg * ns_per_cycle,
        responder_stats.avg * ns_per_cycle,
        net_stats.avg * ns_per_cycle,
        per_hop_latency_stats.avg * ns_per_cycle);
    log_info(
        tt::LogTest,
        "P50       {:>15.2f}    {:>18.2f}    {:>15.2f}    {:>15.2f}",
        raw_stats.p50 * ns_per_cycle,
        responder_stats.p50 * ns_per_cycle,
        net_stats.p50 * ns_per_cycle,
        per_hop_latency_stats.p50 * ns_per_cycle);
    log_info(
        tt::LogTest,
        "P99       {:>15.2f}    {:>18.2f}    {:>15.2f}    {:>15.2f}",
        raw_stats.p99 * ns_per_cycle,
        responder_stats.p99 * ns_per_cycle,
        net_stats.p99 * ns_per_cycle,
        per_hop_latency_stats.p99 * ns_per_cycle);
    log_info(tt::LogTest, "========================================================================");
    log_info(tt::LogTest, "");

    // Populate LatencyResultSummary structure for CSV export
    LatencyResultSummary latency_summary;
    latency_summary.test_name = config.parametrized_name;

    // Extract ftype and ntype from first sender's first pattern
    const TrafficPatternConfig& first_pattern = fetch_first_traffic_pattern(config);
    latency_summary.ftype = fetch_pattern_ftype(first_pattern);
    latency_summary.ntype = fetch_pattern_ntype(first_pattern);

    latency_summary.topology = enchantum::to_string(config.fabric_setup.topology);
    latency_summary.num_devices = test_devices.size();
    latency_summary.num_links = config.fabric_setup.num_links;
    latency_summary.num_samples = raw_latencies_cycles.size();
    latency_summary.payload_size = payload_size;

    // Net latency statistics (most important)
    latency_summary.net_min_ns = net_stats.min * ns_per_cycle;
    latency_summary.net_max_ns = net_stats.max * ns_per_cycle;
    latency_summary.net_avg_ns = net_stats.avg * ns_per_cycle;
    latency_summary.net_p99_ns = net_stats.p99 * ns_per_cycle;

    // Responder processing time statistics
    latency_summary.responder_min_ns = responder_stats.min * ns_per_cycle;
    latency_summary.responder_max_ns = responder_stats.max * ns_per_cycle;
    latency_summary.responder_avg_ns = responder_stats.avg * ns_per_cycle;
    latency_summary.responder_p99_ns = responder_stats.p99 * ns_per_cycle;

    // Raw latency statistics
    latency_summary.raw_min_ns = raw_stats.min * ns_per_cycle;
    latency_summary.raw_max_ns = raw_stats.max * ns_per_cycle;
    latency_summary.raw_avg_ns = raw_stats.avg * ns_per_cycle;
    latency_summary.raw_p99_ns = raw_stats.p99 * ns_per_cycle;

    // Per-hop latency statistics
    latency_summary.per_hop_min_ns = per_hop_latency_stats.min * ns_per_cycle;
    latency_summary.per_hop_max_ns = per_hop_latency_stats.max * ns_per_cycle;
    latency_summary.per_hop_avg_ns = per_hop_latency_stats.avg * ns_per_cycle;
    latency_summary.per_hop_p99_ns = per_hop_latency_stats.p99 * ns_per_cycle;

    // Add to results_summary_ vector (not results_)
    results_summary_.push_back(latency_summary);
}

void LatencyResultsManager::initialize_results_csv_file(bool telemetry_enabled_) {
    // Create output directory
    std::filesystem::path tt_metal_home =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir());
    std::filesystem::path latency_results_path = tt_metal_home / std::string(OUTPUT_DIR);

    if (!std::filesystem::exists(latency_results_path)) {
        std::filesystem::create_directories(latency_results_path);
    }

    // Generate CSV filename (similar to bandwidth summary)
    // Note: The actual file will be created in generate_latency_results_csv() after golden is loaded,
    // similar to how bandwidth_summary_results_*.csv is created in generate_bandwidth_summary_csv()
    auto arch_name = tt::tt_metal::hal::get_arch_name();
    std::ostringstream oss;
    oss << get_perf_metric_name() + "_results_" << arch_name << ".csv";
    csv_file_path_ = latency_results_path / oss.str();

    // Create detailed CSV file with header
    std::ofstream csv_stream(csv_file_path_, std::ios::out | std::ios::trunc);  // Truncate file
    if (!csv_stream.is_open()) {
        log_error(tt::LogTest, "Failed to create {} CSV file: {}", get_perf_metric_name(), csv_file_path_.string());
        return;
    }

    // Write header
    csv_stream << "test_name,ftype,ntype,topology,num_devices,num_links,num_samples,payload_size,"
                  "net_min_ns,net_max_ns,net_avg_ns,net_p99_ns,"
                  "responder_min_ns,responder_max_ns,responder_avg_ns,responder_p99_ns,"
                  "raw_min_ns,raw_max_ns,raw_avg_ns,raw_p99_ns,"
                  "per_hop_min_ns,per_hop_max_ns,per_hop_avg_ns,per_hop_p99_ns,tolerance_percent\n";

    csv_stream.close();

    log_info(tt::LogTest, "Initialized {} CSV file path: {}", get_perf_metric_name(), csv_file_path_.string());
}

void LatencyResultsManager::write_summary_csv_to_file(
    const std::filesystem::path& csv_path, bool include_upload_columns) {
    std::ofstream csv_stream(csv_path, std::ios::out | std::ios::trunc);
    if (!csv_stream.is_open()) {
        log_error(tt::LogTest, "Failed to create CSV file: {}", csv_path.string());
        return;
    }

    if (include_upload_columns) {
        csv_stream << "file_name,machine_type,test_ts,";
    }
    csv_stream << "test_name,ftype,ntype,topology,num_devices,num_links,num_samples,payload_size,"
                  "net_min_ns,net_max_ns,net_avg_ns,net_p99_ns,"
                  "responder_min_ns,responder_max_ns,responder_avg_ns,responder_p99_ns,"
                  "raw_min_ns,raw_max_ns,raw_avg_ns,raw_p99_ns,"
                  "per_hop_min_ns,per_hop_max_ns,per_hop_avg_ns,per_hop_p99_ns,tolerance_percent\n";
    log_info(tt::LogTest, "Initialized latency CSV file: {}", csv_path.string());

    // Write all results
    for (const auto& result : results_summary_) {
        if (include_upload_columns) {
            csv_stream << result.file_name.value() << "," << result.machine_type.value() << ","
                       << result.test_ts.value() << ",";
        }

        csv_stream << result.test_name << "," << result.ftype << "," << result.ntype << "," << result.topology << ","
                   << result.num_devices << "," << result.num_links << "," << result.num_samples << ","
                   << result.payload_size << "," << std::fixed << std::setprecision(2) << result.net_min_ns << ","
                   << result.net_max_ns << "," << result.net_avg_ns << "," << result.net_p99_ns << ","
                   << result.responder_min_ns << "," << result.responder_max_ns << "," << result.responder_avg_ns << ","
                   << result.responder_p99_ns << "," << result.raw_min_ns << "," << result.raw_max_ns << ","
                   << result.raw_avg_ns << "," << result.raw_p99_ns << "," << result.per_hop_min_ns << ","
                   << result.per_hop_max_ns << "," << result.per_hop_avg_ns << "," << result.per_hop_p99_ns << ",";

        // Find the corresponding golden entry for tolerance (like bandwidth does)
        auto golden_it = std::find_if(
            golden_latency_entries_.begin(), golden_latency_entries_.end(), [&](const GoldenLatencyEntry& golden) {
                return golden.test_name == result.test_name && golden.ftype == result.ftype &&
                       golden.ntype == result.ntype && golden.topology == result.topology &&
                       golden.num_devices == result.num_devices && golden.num_links == result.num_links &&
                       golden.payload_size == result.payload_size;
            });

        if (golden_it == golden_latency_entries_.end()) {
            log_warning(
                tt::LogTest,
                "Golden latency entry not found for test {}, putting tolerance of 1.0 in CSV",
                result.test_name);
            csv_stream << 1.0;
        } else {
            csv_stream << golden_it->tolerance_percent;
        }
        csv_stream << "\n";
    }

    csv_stream.close();
    log_info(tt::LogTest, "Latency results written to CSV file: {}", csv_file_path_.string());
}

void LatencyResultsManager::load_golden_csv() {
    golden_latency_entries_.clear();

    std::string golden_filename = get_golden_csv_filename();
    std::filesystem::path golden_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_metal/perf_microbenchmark/routing/golden" / golden_filename;

    if (!std::filesystem::exists(golden_path)) {
        log_warning(tt::LogTest, "Golden latency CSV file not found: {}", golden_path.string());
        return;
    }

    std::ifstream golden_file(golden_path);
    if (!golden_file.is_open()) {
        log_error(tt::LogTest, "Failed to open golden latency CSV file: {}", golden_path.string());
        return;
    }

    std::string line;
    bool is_header = true;
    while (std::getline(golden_file, line)) {
        if (is_header) {
            is_header = false;
            continue;  // Skip header
        }

        std::istringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;

        // Parse CSV line
        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }

        // Expected format: test_name,ftype,ntype,topology,num_devices,num_links,num_samples,payload_size,
        //                  net_min_ns,net_max_ns,net_avg_ns,net_p99_ns,
        //                  responder_min_ns,responder_max_ns,responder_avg_ns,responder_p99_ns,
        //                  raw_min_ns,raw_max_ns,raw_avg_ns,raw_p99_ns,
        //                  per_hop_min_ns,per_hop_max_ns,per_hop_avg_ns,per_hop_p99_ns[,tolerance_percent]
        // Note: per_hop fields and tolerance_percent are optional for backward compatibility
        if (tokens.size() < 20) {
            log_error(
                tt::LogTest,
                "Invalid CSV format in golden latency file. Expected at least 20 fields, got {}",
                tokens.size());
            continue;
        }

        GoldenLatencyEntry entry;
        entry.test_name = tokens[0];
        entry.ftype = tokens[1];
        entry.ntype = tokens[2];
        entry.topology = tokens[3];
        entry.num_devices = std::stoul(tokens[4]);
        entry.num_links = std::stoul(tokens[5]);
        entry.num_samples = std::stoul(tokens[6]);
        entry.payload_size = std::stoul(tokens[7]);

        entry.net_min_ns = std::stod(tokens[8]);
        entry.net_max_ns = std::stod(tokens[9]);
        entry.net_avg_ns = std::stod(tokens[10]);
        entry.net_p99_ns = std::stod(tokens[11]);

        entry.responder_min_ns = std::stod(tokens[12]);
        entry.responder_max_ns = std::stod(tokens[13]);
        entry.responder_avg_ns = std::stod(tokens[14]);
        entry.responder_p99_ns = std::stod(tokens[15]);

        entry.raw_min_ns = std::stod(tokens[16]);
        entry.raw_max_ns = std::stod(tokens[17]);
        entry.raw_avg_ns = std::stod(tokens[18]);
        entry.raw_p99_ns = std::stod(tokens[19]);

        // Per-hop fields are optional for backward compatibility
        if (tokens.size() >= 24) {
            entry.per_hop_min_ns = std::stod(tokens[20]);
            entry.per_hop_max_ns = std::stod(tokens[21]);
            entry.per_hop_avg_ns = std::stod(tokens[22]);
            entry.per_hop_p99_ns = std::stod(tokens[23]);
        } else {
            // If per-hop fields are missing, set to 0
            entry.per_hop_min_ns = 0.0;
            entry.per_hop_max_ns = 0.0;
            entry.per_hop_avg_ns = 0.0;
            entry.per_hop_p99_ns = 0.0;
        }

        // Tolerance is optional for backward compatibility
        if (tokens.size() >= 25) {
            entry.tolerance_percent = std::stod(tokens[24]);
        } else if (tokens.size() >= 21 && tokens.size() < 24) {
            // Old format: tolerance is at position 20
            entry.tolerance_percent = std::stod(tokens[20]);
        } else {
            entry.tolerance_percent = 10.0;  // Default tolerance if not specified
        }
        golden_latency_entries_.push_back(entry);
    }

    golden_file.close();
    log_info(
        tt::LogTest, "Loaded {} golden latency entries from: {}", golden_latency_entries_.size(), golden_path.string());
}

void LatencyResultsManager::compare_latency_results_with_golden() {
    if (golden_latency_entries_.empty()) {
        log_warning(tt::LogTest, "Skipping golden latency comparison - no golden file found");
        return;
    }
    if (results_.size() != golden_latency_entries_.size()) {
        log_warning(
            tt::LogTest,
            "Number of latency results ({}) does not match number of golden entries ({})",
            results_.size(),
            golden_latency_entries_.size());
    }

    for (const auto& test_result : results_) {
        auto golden_it = std::find_if(
            golden_latency_entries_.begin(), golden_latency_entries_.end(), [&](const GoldenLatencyEntry& golden) {
                return golden.test_name == test_result.test_name && golden.ftype == test_result.ftype &&
                       golden.ntype == test_result.ntype && golden.topology == test_result.topology &&
                       golden.num_devices == test_result.num_devices && golden.num_links == test_result.num_links &&
                       golden.payload_size == test_result.payload_size;
            });

        // Create comparison result
        LatencyComparisonResult comp_result;
        comp_result.test_name = test_result.test_name;
        comp_result.ftype = test_result.ftype;
        comp_result.ntype = test_result.ntype;
        comp_result.topology = test_result.topology;
        comp_result.num_devices = test_result.num_devices;
        comp_result.num_links = test_result.num_links;
        comp_result.num_samples = test_result.num_samples;
        comp_result.payload_size = test_result.payload_size;
        comp_result.current_per_hop_avg_ns = test_result.per_hop_avg_ns;

        // Populate golden value and tolerance/status using common helper
        if (golden_it != golden_latency_entries_.end()) {
            comp_result.golden_per_hop_avg_ns = golden_it->per_hop_avg_ns;
        } else {
            comp_result.golden_per_hop_avg_ns = 0.0;
        }
        populate_comparison_tolerance_and_status(comp_result, golden_it, golden_latency_entries_.end());

        latency_comparison_results_.push_back(comp_result);

        // Only count as failure if golden entry exists and test failed
        // NO_GOLDEN status is just a warning, not a failure
        if (!comp_result.within_tolerance && comp_result.status != "NO_GOLDEN") {
            std::ostringstream oss;
            oss << comp_result.test_name << " [" << comp_result.status << "]";
            all_failed_latency_tests_.push_back(oss.str());
        }
    }

    // Initialize diff CSV file using common helper
    auto diff_csv_stream = init_diff_csv_file(
        diff_csv_file_path_,
        "test_name,ftype,ntype,topology,num_devices,num_links,num_samples,payload_size,"
        "current_per_hop_avg_ns,golden_per_hop_avg_ns,difference_percent,status",
        "latency");

    if (diff_csv_stream.is_open()) {
        for (const auto& result : latency_comparison_results_) {
            diff_csv_stream << result.test_name << "," << result.ftype << "," << result.ntype << "," << result.topology
                            << "," << result.num_devices << "," << result.num_links << "," << result.num_samples << ","
                            << result.payload_size << "," << std::fixed << std::setprecision(2)
                            << result.current_per_hop_avg_ns << "," << result.golden_per_hop_avg_ns << ","
                            << result.difference_percent() << "," << result.status << "\n";
        }
        diff_csv_stream.close();
        log_info(tt::LogTest, "Latency comparison diff CSV results written to: {}", diff_csv_file_path_.string());
    }
}

void LatencyResultsManager::validate_against_golden() {
    bool has_latency_results = !latency_comparison_results_.empty();

    // Report latency failures separately
    if (has_latency_results) {
        if (!all_failed_latency_tests_.empty()) {
            has_failures_ = true;
            log_error(tt::LogTest, "=== LATENCY TEST FAILURES ===");
            log_error(
                tt::LogTest,
                "{} latency test(s) failed golden comparison (using per-test tolerance):",
                all_failed_latency_tests_.size());

            // Print detailed failure information
            for (const auto& result : latency_comparison_results_) {
                if (!result.within_tolerance && result.status != "NO_GOLDEN") {
                    // Look up tolerance from golden entry by searching directly
                    double tolerance = 1.0;
                    for (const auto& golden : golden_latency_entries_) {
                        if (golden.test_name == result.test_name && golden.ftype == result.ftype &&
                            golden.ntype == result.ntype && golden.topology == result.topology &&
                            golden.num_devices == result.num_devices && golden.num_links == result.num_links &&
                            golden.payload_size == result.payload_size) {
                            tolerance = golden.tolerance_percent;
                            break;
                        }
                    }

                    log_error(tt::LogTest, "  - {} [{}]:", result.test_name, result.status);
                    log_error(
                        tt::LogTest,
                        "      Test config: {} {} {} {} devs {} links payload={}B",
                        result.ftype,
                        result.ntype,
                        result.topology,
                        result.num_devices,
                        result.num_links,
                        result.payload_size);
                    log_error(tt::LogTest, "      Expected per-hop: {:.2f} ns", result.golden_per_hop_avg_ns);
                    log_error(tt::LogTest, "      Actual per-hop:   {:.2f} ns", result.current_per_hop_avg_ns);
                    log_error(
                        tt::LogTest,
                        "      Diff:     {:.2f}% (tolerance: {:.2f}%)",
                        result.difference_percent(),
                        tolerance);
                }
            }
        } else {
            log_info(
                tt::LogTest,
                "All {} latency tests passed golden comparison using per-test tolerance values",
                latency_comparison_results_.size());
        }
    }
}

void LatencyResultsManager::generate_summary() {
    // Generate latency results CSV file with all results
    generate_summary_csv();
    generate_summary_upload_csv();

    // Compare latency results with golden CSV
    compare_latency_results_with_golden();

    // Validate latency results against golden
    validate_against_golden();
}

void LatencyResultsManager::setup_ci_artifacts() {
    std::filesystem::path tt_metal_home =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir());
    std::filesystem::path ci_artifacts_path = tt_metal_home / std::string(CI_ARTIFACTS_DIR);
    if (!std::filesystem::exists(ci_artifacts_path)) {
        try {
            std::filesystem::create_directories(ci_artifacts_path);
        } catch (const std::filesystem::filesystem_error& e) {
            log_error(
                tt::LogTest, "Failed to create CI artifacts directory, skipping CI artifacts creation: {}", e.what());
            return;
        }
    }

    // Latency artifacts
    for (const std::filesystem::path& csv_filepath : {csv_file_path_, diff_csv_file_path_}) {
        if (csv_filepath.empty()) {
            continue;
        }
        try {
            std::filesystem::copy_file(
                csv_filepath,
                ci_artifacts_path / csv_filepath.filename(),
                std::filesystem::copy_options::overwrite_existing);
        } catch (const std::filesystem::filesystem_error& e) {
            log_debug(
                tt::LogTest,
                "Failed to copy CSV file {} to CI artifacts directory: {}",
                csv_filepath.filename().string(),
                e.what());
        }
    }
    log_trace(tt::LogTest, "Copied latency CSV files to CI artifacts directory: {}", ci_artifacts_path.string());
}

void LatencyResultsManager::reset_state() {
    latency_comparison_results_.clear();
    all_failed_latency_tests_.clear();
    has_failures_ = false;
}

template class ResultsManager<BandwidthResult, BandwidthResultSummary>;
template class ResultsManager<LatencyResult, LatencyResultSummary>;

}  // namespace tt::tt_fabric::fabric_tests
