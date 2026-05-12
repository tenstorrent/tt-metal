// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace ttml::benchmark_utils {

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

inline std::vector<uint32_t> parse_u32_csv(std::string_view csv) {
    std::vector<uint32_t> out;
    std::stringstream ss{std::string(csv)};
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            continue;
        }
        out.push_back(static_cast<uint32_t>(std::stoul(token)));
    }
    return out;
}

inline std::vector<std::string> parse_string_csv(std::string_view csv) {
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

inline bool name_is_enabled(const std::vector<std::string>& enabled_names, const std::string& name) {
    return enabled_names.empty() || std::find(enabled_names.begin(), enabled_names.end(), name) != enabled_names.end();
}

inline uint32_t seed_from_name(const std::string& name) {
    return static_cast<uint32_t>(std::hash<std::string>{}(name));
}

}  // namespace ttml::benchmark_utils
