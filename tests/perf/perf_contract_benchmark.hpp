// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Google Benchmark side of the runtime perf output contract. See perf_contract.hpp.

#pragma once

#include <string>
#include <string_view>

#include <benchmark/benchmark.h>

#include "perf/perf_contract.hpp"

namespace tt::perf {

// Declares a gated metric. Call before benchmark::RunSpecifiedBenchmarks(); every case must report a counter
// (or built-in field such as bytes_per_second) with this name.
inline void declare_metric(std::string_view counter, const MetricDecl& decl) {
    auto [key, value] = metric_context_entry(counter, decl);
    benchmark::AddCustomContext(key, value);
}

// Records run-wide context such as IOMMU state. Values are shown next to the comparison so a machine change
// is not mistaken for a code change.
inline void add_run_context(std::string_view key, std::string_view value) {
    benchmark::AddCustomContext(fmt::format("perf.context.{}", key), std::string(value));
}

// Records per-case context that is only meaningful while the device is open, such as AICLK.
inline void add_case_context(benchmark::State& state, std::string_view key, double value) {
    state.counters[fmt::format("ctx_{}", key)] = benchmark::Counter(value, benchmark::Counter::kDefaults);
}

}  // namespace tt::perf
