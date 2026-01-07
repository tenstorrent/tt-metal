// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <vector>
#include <fstream>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include "tt_fabric_telemetry.hpp"
#include "tt_fabric_test_eth_readback.hpp"
#include "tt_fabric_test_constants.hpp"
#include "tt_fabric_test_common_types.hpp"

using tt::tt_fabric::fabric_tests::OUTPUT_DIR;

using TestConfig = tt::tt_fabric::fabric_tests::TestConfig;
using TrafficPatternConfig = tt::tt_fabric::fabric_tests::TrafficPatternConfig;

using MeshCoordinate = tt::tt_metal::distributed::MeshCoordinate;

// Helper functions for code profiling
using tt::tt_fabric::convert_code_profiling_timer_type_to_str;
using tt::tt_fabric::convert_to_code_profiling_timer_type;

// Helper functions for parsing traffic pattern parameters
using tt::tt_fabric::fabric_tests::fetch_first_traffic_pattern;
using tt::tt_fabric::fabric_tests::fetch_pattern_ftype;
using tt::tt_fabric::fabric_tests::fetch_pattern_ntype;
using tt::tt_fabric::fabric_tests::fetch_pattern_packet_size;

// Manages fabric code profiling lifecycle (readback, clearing, reporting).
class CodeProfiler {
public:
    explicit CodeProfiler(EthCoreBufferReadback& eth_readback);

    void set_enabled(bool enabled);
    bool is_enabled() const;

    void clear_code_profiling_buffers();
    void read_code_profiling_results();
    void report_code_profiling_results() const;
    void initialize_code_profiling_results_csv_file();
    std::string convert_coord_to_string(const MeshCoordinate& coord);
    void dump_code_profiling_results_to_csv(const TestConfig& config);
    void reset();  // Clears entries and device buffers when profiling is enabled

    const std::vector<CodeProfilingEntry>& get_entries() const;

private:
    EthCoreBufferReadback& eth_readback_;
    std::vector<CodeProfilingEntry> entries_;
    bool enabled_ = false;
    std::filesystem::path code_profiling_csv_file_path_;
};
