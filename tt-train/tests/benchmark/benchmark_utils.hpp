// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace ttml::benchmark_utils {

struct BenchmarkIterationConfig {
    int num_warmup_iterations = 0;
    int num_measurement_iterations = 0;
};

// Benchmark-only sweep helpers.
//
// Keep these separate from test_utils: test_utils owns reusable test data construction,
// while benchmark_utils owns benchmark CLI/env plumbing and reporting conveniences.
// Only place helpers here once they have real benchmark call sites; speculative helpers
// should stay local to the benchmark that needs them.

inline std::string join_u32_csv(const std::vector<uint32_t>& values) {
    std::string out;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out += ",";
        }
        out += std::to_string(values[i]);
    }
    return out;
}

inline std::string_view trim_ascii_whitespace(std::string_view value) {
    constexpr std::string_view kWhitespace = " \t\n\r\f\v";
    const auto begin = value.find_first_not_of(kWhitespace);
    if (begin == std::string_view::npos) {
        return {};
    }
    const auto end = value.find_last_not_of(kWhitespace);
    return value.substr(begin, end - begin + 1U);
}

inline uint32_t parse_u32_strict(const std::string_view raw_token) {
    const auto token = trim_ascii_whitespace(raw_token);
    if (token.empty()) {
        throw std::invalid_argument("Expected a non-empty uint32 token.");
    }

    uint64_t parsed_value = 0;
    const auto* begin = token.data();
    const auto* end = token.data() + token.size();
    const auto [ptr, ec] = std::from_chars(begin, end, parsed_value, 10);
    if (ec != std::errc{} || ptr != end || parsed_value > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("Invalid uint32 token: \"" + std::string(token) + "\"");
    }

    return static_cast<uint32_t>(parsed_value);
}

inline std::vector<uint32_t> parse_u32_csv(const std::string_view csv) {
    std::vector<uint32_t> out;
    std::stringstream ss{std::string(csv)};
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (trim_ascii_whitespace(token).empty()) {
            continue;
        }
        out.push_back(parse_u32_strict(token));
    }
    return out;
}

inline std::vector<std::string> parse_string_csv(const std::string_view csv) {
    std::vector<std::string> out;
    std::stringstream ss{std::string(csv)};
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            continue;
        }
        out.push_back(token);
    }
    return out;
}

inline void override_u32_from_env(const char* env_name, uint32_t& value) {
    if (const char* env = std::getenv(env_name)) {
        value = parse_u32_strict(env);
    }
}

inline void override_u32_csv_from_env(const char* env_name, std::vector<uint32_t>& values) {
    if (const char* env = std::getenv(env_name)) {
        auto parsed = parse_u32_csv(env);
        if (!parsed.empty()) {
            values = std::move(parsed);
        }
    }
}

inline void override_string_csv_from_env(const char* env_name, std::vector<std::string>& values) {
    if (const char* env = std::getenv(env_name)) {
        values = parse_string_csv(env);
    }
}

inline bool name_is_enabled(const std::vector<std::string>& enabled_names, const std::string& name) {
    return enabled_names.empty() || std::find(enabled_names.begin(), enabled_names.end(), name) != enabled_names.end();
}

inline uint32_t seed_from_name(const std::string& name) {
    return static_cast<uint32_t>(std::hash<std::string>{}(name));
}

// Relative change from reference (%). Positive means value increased.
inline double relative_change_pct(const double value, const double reference) {
    if (reference == 0.0) {
        return 0.0;
    }
    return (value - reference) / reference * 100.0;
}

// Reduction against baseline (%). Positive means value decreased.
inline double reduction_pct(const double baseline, const double value) {
    return -relative_change_pct(value, baseline);
}

inline double speedup_x(const double baseline, const double value) {
    if (value == 0.0) {
        return 0.0;
    }
    return baseline / value;
}

inline double average(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

template <typename Fn>
inline double measure_average_iteration_time_s(const int num_iterations, Fn&& fn) {
    if (num_iterations <= 0) {
        return 0.0;
    }
    auto total_time = std::chrono::duration<double>::zero();
    for (int iter = 0; iter < num_iterations; ++iter) {
        const auto start = std::chrono::high_resolution_clock::now();
        fn();
        const auto end = std::chrono::high_resolution_clock::now();
        total_time += end - start;
    }
    return total_time.count() / static_cast<double>(num_iterations);
}

}  // namespace ttml::benchmark_utils
