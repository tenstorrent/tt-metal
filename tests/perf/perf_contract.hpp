// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Output contract for runtime performance benchmarks consumed by tests/perf.
//
// A benchmark declares each gated metric once (unit, direction, aggregation) and reports one value per
// repetition for every case. Case names use key:value segments for swept arguments so the consumer can group
// them without knowing the benchmark. Google Benchmark binaries meet the contract through
// perf_contract_benchmark.hpp; other binaries write the same JSON shape with write_result().

#pragma once

#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>

namespace tt::perf {

enum class Better { Lower, Higher };
enum class Aggregate { Min, Median, Max };

struct MetricDecl {
    std::string unit;
    Better better;
    Aggregate aggregate;
};

inline std::string_view to_string(Better better) { return better == Better::Lower ? "lower" : "higher"; }

inline std::string_view to_string(Aggregate aggregate) {
    switch (aggregate) {
        case Aggregate::Min: return "min";
        case Aggregate::Median: return "median";
        case Aggregate::Max: return "max";
    }
    return "min";
}

// Context key and value that declare a metric in the JSON "context" block.
inline std::pair<std::string, std::string> metric_context_entry(std::string_view counter, const MetricDecl& decl) {
    return {
        fmt::format("perf.metric.{}", counter),
        fmt::format("unit={};better={};aggregate={}", decl.unit, to_string(decl.better), to_string(decl.aggregate))};
}

struct CaseResult {
    std::string name;
    std::map<std::string, double> counters;
};

namespace detail {
inline std::string json_escape(std::string_view s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') {
            out += '\\';
        }
        out += c;
    }
    return out;
}
}  // namespace detail

// Writes results in the Google Benchmark JSON shape for binaries that are not Google Benchmark executables.
// Each call describes one repetition; the consumer merges repetitions from separate processes.
inline void write_result(
    const std::string& path,
    const std::vector<std::pair<std::string, MetricDecl>>& metrics,
    const std::vector<CaseResult>& cases) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error(fmt::format("cannot open {} for writing perf result", path));
    }
    out << "{\n  \"context\": {";
    for (size_t i = 0; i < metrics.size(); ++i) {
        auto [key, value] = metric_context_entry(metrics[i].first, metrics[i].second);
        out << (i ? ",\n" : "\n") << fmt::format(R"(    "{}": "{}")", key, value);
    }
    out << "\n  },\n  \"benchmarks\": [";
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto name = detail::json_escape(cases[i].name);
        out << (i ? ",\n" : "\n")
            << fmt::format(
                   R"(    {{"name": "{0}", "run_name": "{0}", "run_type": "iteration", "repetitions": 1)", name);
        for (const auto& [counter, value] : cases[i].counters) {
            out << fmt::format(R"(, "{}": {:.17g})", detail::json_escape(counter), value);
        }
        out << "}";
    }
    out << "\n  ]\n}\n";
    if (!out) {
        throw std::runtime_error(fmt::format("failed to write perf result {}", path));
    }
}

}  // namespace tt::perf
