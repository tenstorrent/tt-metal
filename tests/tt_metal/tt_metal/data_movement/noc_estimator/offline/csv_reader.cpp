// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "csv_reader.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

namespace tt::noc_estimator::offline {

static std::string to_lower(const std::string& str) {
    std::string result = str;
    transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

bool CsvReader::load_csv(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open CSV file: " << filepath << std::endl;
        return false;
    }

    data_points_.clear();
    column_map_.clear();

    std::string line;
    bool first_line = true;

    while (getline(file, line)) {
        // Skip empty lines
        if (line.empty()) {
            continue;
        }

        // Parse header on first line
        if (first_line) {
            first_line = false;
            if (!parse_header(line)) {
                std::cerr << "Failed to parse CSV header" << std::endl;
                return false;
            }
            continue;
        }

        // Parse data lines
        DataPoint point;
        if (parse_data_line(line, point)) {
            data_points_.push_back(point);
        }
    }

    return !data_points_.empty();
}

const std::vector<DataPoint>& CsvReader::get_data_points() const { return data_points_; }

std::vector<std::string> CsvReader::split_csv_line(const std::string& line) {
    std::stringstream ss(line);
    std::string token;
    std::vector<std::string> tokens;

    while (getline(ss, token, ',')) {
        // Trim whitespace
        size_t start = token.find_first_not_of(" \t\r\n");
        size_t end = token.find_last_not_of(" \t\r\n");
        if (start != std::string::npos) {
            tokens.push_back(token.substr(start, end - start + 1));
        } else {
            tokens.push_back("");
        }
    }

    return tokens;
}

bool CsvReader::parse_header(const std::string& line) {
    std::vector<std::string> headers = split_csv_line(line);

    for (size_t i = 0; i < headers.size(); i++) {
        column_map_[headers[i]] = i;
    }

    if (!column_map_.contains(COL_TRANSACTION_SIZE) || !column_map_.contains(COL_NUM_TRANSACTIONS) ||
        !column_map_.contains(COL_LATENCY)) {
        std::cerr << "CSV missing required columns" << std::endl;
        return false;
    }

    return true;
}

bool CsvReader::parse_data_line(const std::string& line, DataPoint& point) {
    std::vector<std::string> tokens = split_csv_line(line);

    if (tokens.empty()) {
        return false;
    }

    try {
        // Required fields
        point.transaction_size_bytes = stoul(tokens[column_map_[COL_TRANSACTION_SIZE]]);
        point.num_transactions = stoul(tokens[column_map_[COL_NUM_TRANSACTIONS]]);
        point.latency_cycles = stod(tokens[column_map_[COL_LATENCY]]);
        auto it = column_map_.find(COL_NUM_SUBORDINATES);
        if (it != column_map_.end() && it->second < tokens.size()) {
            point.num_subordinates = stoul(tokens[it->second]);
        }

        it = column_map_.find(COL_SAME_AXIS);
        if (it != column_map_.end() && it->second < tokens.size()) {
            point.same_axis = (stoi(tokens[it->second]) != 0);
        }

        it = column_map_.find(COL_LINKED);
        if (it != column_map_.end() && it->second < tokens.size()) {
            std::string linked_lower = to_lower(tokens[it->second]);
            point.linked = (linked_lower == "true");
        }

        it = column_map_.find(COL_ARCH);
        if (it != column_map_.end() && it->second < tokens.size()) {
            std::string arch_lower = to_lower(tokens[it->second]);
            if (arch_lower == "blackhole") {
                point.arch = common::Architecture::BLACKHOLE;
            }
        }

        it = column_map_.find(COL_MECHANISM);
        if (it != column_map_.end() && it->second < tokens.size()) {
            std::string mech_lower = to_lower(tokens[it->second]);
            if (mech_lower == "multicast") {
                point.mechanism = common::NocMechanism::MULTICAST;
            }
        }

        it = column_map_.find(COL_MEMORY);
        if (it != column_map_.end() && it->second < tokens.size()) {
            std::string mem_lower = to_lower(tokens[it->second]);
            if (mem_lower == "dram") {
                point.memory = common::MemoryType::DRAM;
            }
        }

        it = column_map_.find(COL_PATTERN);
        if (it != column_map_.end() && it->second < tokens.size()) {
            std::string pattern_lower = to_lower(tokens[it->second]);
            if (pattern_lower == "one_from_one") {
                point.pattern = common::NocPattern::ONE_FROM_ONE;
            } else if (pattern_lower == "one_to_one") {
                point.pattern = common::NocPattern::ONE_TO_ONE;
            } else if (pattern_lower == "one_to_all") {
                point.pattern = common::NocPattern::ONE_TO_ALL;
            } else if (pattern_lower == "one_from_all") {
                point.pattern = common::NocPattern::ONE_FROM_ALL;
            } else if (pattern_lower == "all_to_all") {
                point.pattern = common::NocPattern::ALL_TO_ALL;
            } else if (pattern_lower == "all_from_all") {
                point.pattern = common::NocPattern::ALL_FROM_ALL;
            }
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse data line: " << line << " | error: " << e.what() << std::endl;
        return false;
    }
}

}  // namespace tt::noc_estimator::offline
