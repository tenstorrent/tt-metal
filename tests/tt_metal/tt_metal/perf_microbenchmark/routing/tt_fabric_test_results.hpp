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
#include <map>
#include <cmath>
#include <enchantum/enchantum.hpp>
#include "tt_fabric_test_config.hpp"
#include "tt_metal/fabric/fabric_edm_packet_header.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include "tt_fabric_test_common_types.hpp"
#include "tt_fabric_test_common.hpp"
#include "tt_fabric_test_interfaces.hpp"
#include "tt_fabric_test_memory_map.hpp"
#include "tt_fabric_test_device_setup.hpp"
#include "tt_fabric_test_traffic.hpp"
#include "tt_fabric_test_constants.hpp"

namespace tt::tt_fabric::fabric_tests {

//=============================ENUMS==========================

// Measurement type
enum class PerformanceMetric { Bandwidth = 0, Latency = 1 };

const std::unordered_map<PerformanceMetric, std::string> PerformanceMetricString{
    {PerformanceMetric::Bandwidth, "bandwidth"},
    {PerformanceMetric::Latency, "latency"},
};
// Aggregates bandwidth results, generates CSVs, and compares against golden data.
template <typename T, typename U>
class ResultsManager {
protected:
    PerformanceMetric performance_t{};
    std::vector<T> results_;
    std::vector<U> results_summary_;
    std::filesystem::path csv_summary_file_path_;
    std::filesystem::path csv_summary_upload_file_path_;
    std::filesystem::path csv_file_path_;
    std::filesystem::path diff_csv_file_path_;

public:
    std::string get_perf_metric_name() const { return PerformanceMetricString.at(performance_t); }
    std::string get_golden_csv_filename();
    void generate_summary_csv();
    void populate_upload_metadata_fields();
    void generate_summary_upload_csv();
    std::string convert_num_devices_to_string(const std::vector<uint32_t>& num_devices) const;
    std::ofstream init_diff_csv_file(
        std::filesystem::path& diff_csv_path, const std::string& csv_header, const std::string& test_type);
    virtual void generate_summary() = 0;
    virtual void write_summary_csv_to_file(const std::filesystem::path& csv_path, bool include_upload_columns) = 0;
    virtual void initialize_results_csv_file(bool telemetry_enabled = false) = 0;
    virtual void load_golden_csv() = 0;
    virtual ~ResultsManager() = default;
};

}  // namespace tt::tt_fabric::fabric_tests
