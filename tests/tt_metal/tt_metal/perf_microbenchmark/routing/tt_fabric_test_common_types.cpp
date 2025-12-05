// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/tt_fabric_test_common_types.hpp"

namespace tt::tt_fabric::fabric_tests {

// Helper functions for fetching pattern parameters
TrafficPatternConfig fetch_first_traffic_pattern(const TestConfig& config) {
    TT_FATAL(
        !config.senders.empty() && !config.senders[0].patterns.empty(),
        "No senders or patterns found for test {}",
        config.name);
    return config.senders[0].patterns[0];
}

std::string fetch_pattern_test_type(const TrafficPatternConfig& pattern, auto lambda_test_type) {
    const auto& test_type = lambda_test_type(pattern);
    TT_FATAL(test_type.has_value(), "Test type not found in pattern");
    return std::string(enchantum::to_string(test_type.value()));
}

std::string fetch_pattern_ftype(const TrafficPatternConfig& pattern) {
    log_debug(tt::LogTest, "Fetching ftype from pattern");
    return fetch_pattern_test_type(pattern, [](const auto& pattern) { return pattern.ftype; });
}

std::string fetch_pattern_ntype(const TrafficPatternConfig& pattern) {
    log_debug(tt::LogTest, "Fetching ntype from pattern");
    return fetch_pattern_test_type(pattern, [](const auto& pattern) { return pattern.ntype; });
}

uint32_t fetch_pattern_int(const TrafficPatternConfig& pattern, auto lambda_parameter) {
    const auto& parameter = lambda_parameter(pattern);
    TT_FATAL(parameter.has_value(), "Parameter not found in pattern");
    return parameter.value();
}

uint32_t fetch_pattern_num_packets(const TrafficPatternConfig& pattern) {
    log_debug(tt::LogTest, "Fetching num_packets from pattern");
    return fetch_pattern_int(pattern, [](const auto& pattern) { return pattern.num_packets; });
}

uint32_t fetch_pattern_packet_size(const TrafficPatternConfig& pattern) {
    log_debug(tt::LogTest, "Fetching packet size from pattern");
    return fetch_pattern_int(pattern, [](const auto& pattern) { return pattern.size; });
}

}  // namespace tt::tt_fabric::fabric_tests
