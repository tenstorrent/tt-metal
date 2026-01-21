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

BandwidthResultsManager::BandwidthResultsManager() = default;

void BandwidthResultsManager::initialize_bandwidth_csv_file(bool telemetry_enabled) {
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
    oss << "bandwidth_results_" << arch_name << ".csv";
    csv_file_path_ = bandwidth_results_path / oss.str();

    // Create detailed CSV file with header
    std::ofstream csv_stream(csv_file_path_, std::ios::out | std::ios::trunc);  // Truncate file
    if (!csv_stream.is_open()) {
        log_error(tt::LogTest, "Failed to create CSV file: {}", csv_file_path_.string());
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

    log_info(tt::LogTest, "Initialized CSV file: {}", csv_file_path_.string());
}

void BandwidthResultsManager::add_result(const TestConfig& config, const BandwidthResult& result) {
    (void)config;
    bandwidth_results_.push_back(result);
}

void BandwidthResultsManager::add_summary(const TestConfig& config, const BandwidthResultSummary& summary) {
    const std::string& test_name = config.name;
    // First iteration or first time we see this test name
    if (config.iteration_number == 0 || !test_name_to_summary_index_.contains(test_name)) {
        test_name_to_summary_index_[test_name] = bandwidth_results_summary_.size();
        bandwidth_results_summary_.push_back(summary);
        return;
    }

    // Append to existing summary
    auto idx = test_name_to_summary_index_.at(test_name);
    BandwidthResultSummary& existing = bandwidth_results_summary_.at(idx);

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

    csv_stream << config.name << "," << ftype_str << "," << ntype_str << ","
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

std::string BandwidthResultsManager::get_golden_csv_filename() {
    auto arch_name = tt::tt_metal::hal::get_arch_name();
    auto cluster_type = tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type();

    // Convert cluster type enum to lowercase string
    std::string cluster_name = std::string(enchantum::to_string(cluster_type));
    std::transform(cluster_name.begin(), cluster_name.end(), cluster_name.begin(), ::tolower);

    std::string file_name = "golden_bandwidth_summary_" + arch_name + "_" + cluster_name + ".csv";
    return file_name;
}

void BandwidthResultsManager::generate_summary() {
    calculate_bandwidth_summary_statistics();
    generate_bandwidth_summary_csv();
    generate_bandwidth_summary_upload_csv();
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

std::string BandwidthResultsManager::convert_num_devices_to_string(const std::vector<uint32_t>& num_devices) const {
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
    for (auto& result : bandwidth_results_summary_) {
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
    for (auto& result : bandwidth_results_summary_) {
        result.statistics_vector.push_back(
            *std::min_element(result.bandwidth_vector_GB_s.begin(), result.bandwidth_vector_GB_s.end()));
    }
}

void BandwidthResultsManager::calculate_bandwidth_max() {
    stat_order_.push_back(BandwidthStatistics::BandwidthMax);
    for (auto& result : bandwidth_results_summary_) {
        result.statistics_vector.push_back(
            *std::max_element(result.bandwidth_vector_GB_s.begin(), result.bandwidth_vector_GB_s.end()));
    }
}

void BandwidthResultsManager::calculate_bandwidth_std_dev() {
    stat_order_.push_back(BandwidthStatistics::BandwidthStdDev);
    for (auto& result : bandwidth_results_summary_) {
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

void BandwidthResultsManager::write_bandwidth_summary_csv_to_file(
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

    for (const auto& result : bandwidth_results_summary_) {
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

void BandwidthResultsManager::populate_upload_metadata_fields() {
    auto arch_name = tt::tt_metal::hal::get_arch_name();
    auto cluster_type = tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type();
    std::string machine_type = std::string(enchantum::to_string(cluster_type));
    std::transform(machine_type.begin(), machine_type.end(), machine_type.begin(), ::tolower);

    std::string file_name = "bw_" + arch_name + "_" + machine_type;

    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm timestamp_now{};
    localtime_r(&time_t_now, &timestamp_now);
    std::ostringstream timestamp_oss;
    timestamp_oss << std::put_time(&timestamp_now, "%Y-%m-%d %H:%M:%S");
    std::string test_ts = timestamp_oss.str();

    for (auto& result : bandwidth_results_summary_) {
        result.file_name = file_name;
        result.machine_type = machine_type;
        result.test_ts = test_ts;
    }
}

void BandwidthResultsManager::generate_bandwidth_summary_csv() {
    auto arch_name = tt::tt_metal::hal::get_arch_name();
    std::ostringstream summary_oss;
    summary_oss << "bandwidth_summary_results_" << arch_name << ".csv";

    std::filesystem::path output_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        std::string(OUTPUT_DIR);
    csv_summary_file_path_ = output_path / summary_oss.str();

    write_bandwidth_summary_csv_to_file(csv_summary_file_path_, false);
}

void BandwidthResultsManager::generate_bandwidth_summary_upload_csv() {
    auto arch_name = tt::tt_metal::hal::get_arch_name();
    std::ostringstream upload_oss;
    upload_oss << "bandwidth_summary_results_" << arch_name << "_upload.csv";

    std::filesystem::path output_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        std::string(OUTPUT_DIR);
    csv_summary_upload_file_path_ = output_path / upload_oss.str();

    populate_upload_metadata_fields();
    write_bandwidth_summary_csv_to_file(csv_summary_upload_file_path_, true);
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

std::ofstream BandwidthResultsManager::init_diff_csv_file(
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

void BandwidthResultsManager::compare_summary_results_with_golden() {
    if (golden_csv_entries_.empty()) {
        log_warning(tt::LogTest, "Skipping golden CSV comparison - no golden file found");
        return;
    }
    if (bandwidth_results_summary_.size() != golden_csv_entries_.size()) {
        log_warning(
            tt::LogTest,
            "Number of test results ({}) does not match number of golden entries ({})",
            bandwidth_results_summary_.size(),
            golden_csv_entries_.size());
    }

    for (auto& test_result : bandwidth_results_summary_) {
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

}  // namespace tt::tt_fabric::fabric_tests
