// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <vector>

#include "tt_fabric_telemetry.hpp"
#include "tt_fabric_test_common_types.hpp"
#include "tt_fabric_test_constants.hpp"
#include "tt_fabric_test_eth_readback.hpp"
#include "tt_fabric_test_results.hpp"

using TestConfig = tt::tt_fabric::fabric_tests::TestConfig;
using FabricNodeId = tt::tt_fabric::FabricNodeId;
using tt::tt_fabric::fabric_tests::OUTPUT_DIR;

// Manages fabric bandwidth telemetry collection and processing.
class TelemetryManager {
public:
    TelemetryManager(TestFixture& fixture, EthCoreBufferReadback& eth_readback);

    void read_telemetry();
    void clear_telemetry();
    void process_telemetry_for_golden();
    void dump_raw_telemetry_csv(const TestConfig& config);
    void reset();

    double get_measured_bw_min() const { return measured_bw_min_; }
    double get_measured_bw_avg() const { return measured_bw_avg_; }
    double get_measured_bw_max() const { return measured_bw_max_; }
    const std::vector<TelemetryEntry>& get_entries() const { return telemetry_entries_; }

private:
    TestFixture& fixture_;
    EthCoreBufferReadback& eth_readback_;

    std::vector<TelemetryEntry> telemetry_entries_;
    double measured_bw_min_ = 0.0;
    double measured_bw_avg_ = 0.0;
    double measured_bw_max_ = 0.0;
    std::filesystem::path raw_telemetry_csv_path_;
};
