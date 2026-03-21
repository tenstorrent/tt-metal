// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <map>
#include <tt-metalium/experimental/noc_estimator/types.hpp>

namespace tt::tt_metal::experimental::noc_estimator::offline {

// Represents a single row of performance data from CSV
struct DataPoint {
    NocMechanism mechanism = NocMechanism::UNICAST;
    NocPattern pattern = NocPattern::ONE_TO_ONE;
    MemoryType memory = MemoryType::L1;
    Architecture arch = Architecture::WORMHOLE_B0;
    uint32_t num_transactions = 0;
    uint32_t transaction_size_bytes = 0;
    uint32_t num_subordinates = 1;
    bool same_axis = false;
    bool stateful = false;
    bool loopback = false;
    uint32_t noc_index = 0;
    double latency_cycles = 0.0;
};

// Reads CSV files and stores performance data
class CsvReader {
public:
    // load a CSV file and parse all data points
    bool load_csv(const std::string& filepath);

    // Get all loaded data points
    const std::vector<DataPoint>& get_data_points() const;

private:
    std::vector<DataPoint> data_points_;
    std::map<std::string, size_t> column_map_;  // Column name -> index

    // Canonical column names
    static constexpr const char* COL_MECHANISM = "Mechanism";
    static constexpr const char* COL_PATTERN = "Pattern";
    static constexpr const char* COL_MEMORY = "Memory type";
    static constexpr const char* COL_ARCH = "Architecture";
    static constexpr const char* COL_NUM_TRANSACTIONS = "Number of Transactions";
    static constexpr const char* COL_TRANSACTION_SIZE = "Transaction Size (bytes)";
    static constexpr const char* COL_NUM_SUBORDINATES = "Number of subordinates";
    static constexpr const char* COL_SAME_AXIS = "Same axis";
    static constexpr const char* COL_STATEFUL = "Stateful";
    static constexpr const char* COL_LOOPBACK = "Loopback";
    static constexpr const char* COL_NOC_INDEX = "NOC index";
    static constexpr const char* COL_LATENCY = "Latency (cycles)";

    bool parse_header(const std::string& line);
    bool parse_data_line(const std::string& line, DataPoint& point);
    std::vector<std::string> split_csv_line(const std::string& line);
};

}  // namespace tt::tt_metal::experimental::noc_estimator::offline
