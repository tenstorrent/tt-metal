// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Shared by the four leg benchmarks: the metric summaries, the counter writer, the process
// mesh and the quiet reporter. Each file keeps only its own init_counters and fixture.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/mesh_device.hpp>

namespace leg_bench {

namespace dist = tt::tt_metal::distributed;
namespace mh = tt::tt_metal::distributed::multihost;

// Fail rather than spin: a parked kernel and an unanswered collective are both unbounded.
constexpr auto kStall = std::chrono::seconds(30);

// SkipWithError does not set the exit status; each main() returns this instead.
inline bool g_run_failed = false;

inline void fail(benchmark::State& state, const std::string& why) {
    g_run_failed = true;
    state.SkipWithError(why);
}

// Copied from benchmark_hd_sockets.cpp:159-215 and :518-528; file-local there.
struct LatencySummary {
    double avg_us = 0.0;
    double min_us = 0.0;
    double max_us = 0.0;
    double p50_us = 0.0;
    double p99_us = 0.0;
    double avg_cycles = 0.0;
    uint64_t min_cycles = 0;
    uint64_t max_cycles = 0;
};

// Zeroed rather than fatal on an empty input: a case that measured nothing reports zeros.
inline LatencySummary summarize_latency_cycles(const std::vector<uint64_t>& cycles, double cycles_per_us) {
    if (cycles.empty() || cycles_per_us <= 0.0) {
        return {};
    }
    auto sorted = cycles;
    std::sort(sorted.begin(), sorted.end());
    double avg_c = 0.0;
    for (const uint64_t c : cycles) {
        avg_c += static_cast<double>(c);
    }
    avg_c /= static_cast<double>(cycles.size());

    auto to_us = [&](double c) { return c / cycles_per_us; };
    return {
        .avg_us = to_us(avg_c),
        .min_us = to_us(static_cast<double>(sorted.front())),
        .max_us = to_us(static_cast<double>(sorted.back())),
        .p50_us = to_us(static_cast<double>(sorted[sorted.size() / 2])),
        .p99_us = to_us(static_cast<double>(sorted[(sorted.size() * 99) / 100])),
        .avg_cycles = avg_c,
        .min_cycles = sorted.front(),
        .max_cycles = sorted.back(),
    };
}

// For a leg stamped on the host clock. cycles_per_us only back-fills the cycle columns,
// and is 0 where no device is in the path.
inline LatencySummary summarize_latency_us(const std::vector<double>& us_values, double cycles_per_us) {
    if (us_values.empty()) {
        return {};
    }
    auto sorted = us_values;
    std::sort(sorted.begin(), sorted.end());
    double avg_us = 0.0;
    for (const double v : us_values) {
        avg_us += v;
    }
    avg_us /= static_cast<double>(us_values.size());

    return {
        .avg_us = avg_us,
        .min_us = sorted.front(),
        .max_us = sorted.back(),
        .p50_us = sorted[sorted.size() / 2],
        .p99_us = sorted[(sorted.size() * 99) / 100],
        .avg_cycles = avg_us * cycles_per_us,
        .min_cycles = static_cast<uint64_t>(sorted.front() * cycles_per_us),
        .max_cycles = static_cast<uint64_t>(sorted.back() * cycles_per_us),
    };
}

inline LatencySummary summarize_latency_ns(const std::vector<uint64_t>& ns_values, double cycles_per_us) {
    std::vector<double> us;
    us.reserve(ns_values.size());
    for (const uint64_t v : ns_values) {
        us.push_back(static_cast<double>(v) / 1e3);
    }
    return summarize_latency_us(us, cycles_per_us);
}

// `prefix` is the one addition: analyze_hd_sockets.py reads the unprefixed set.
inline void set_latency_counters(
    benchmark::State& state, const LatencySummary& s, uint64_t num_iterations, const std::string& prefix = "") {
    state.counters[prefix + "num_iterations"] = static_cast<double>(num_iterations);
    state.counters[prefix + "avg_us"] = s.avg_us;
    state.counters[prefix + "min_us"] = s.min_us;
    state.counters[prefix + "max_us"] = s.max_us;
    state.counters[prefix + "p50_us"] = s.p50_us;
    state.counters[prefix + "p99_us"] = s.p99_us;
    state.counters[prefix + "avg_cycles"] = s.avg_cycles;
    state.counters[prefix + "min_cycles"] = static_cast<double>(s.min_cycles);
    state.counters[prefix + "max_cycles"] = static_cast<double>(s.max_cycles);
}

inline double us_since(std::chrono::steady_clock::time_point t) {
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t).count();
    return static_cast<double>(ns) / 1e3;
}

// The word the kernels write into every payload word but word 0, keyed by core or rank.
inline uint32_t pattern_word(uint32_t index) {
    const uint32_t b = 0x40u + (index & 0x1Fu);
    return b | (b << 8) | (b << 16) | (b << 24);
}

// One bringup per process: Fixture::SetUp runs once per arg case, so a SetUp that built the
// mesh would re-pay bringup at every point of a sweep. First call fixes the device.
inline std::shared_ptr<dist::MeshDevice> unit_mesh(int device_id) {
    static std::shared_ptr<dist::MeshDevice> mesh = dist::MeshDevice::create_unit_mesh(device_id);
    return mesh;
}

// The split keeps each rank to its own device: a unit mesh opened against the full world
// has every rank claiming every chip.
inline std::shared_ptr<dist::MeshDevice> split_unit_mesh(int device_id) {
    static std::shared_ptr<dist::MeshDevice> mesh = [device_id] {
        const mh::ContextPtr world = mh::DistributedContext::get_current_world();
        const mh::ContextPtr solo = world->split(mh::Color{static_cast<int>(*world->rank())}, mh::Key{0});
        mh::DistributedContext::set_current_world(solo);
        auto m = dist::MeshDevice::create_unit_mesh(device_id);
        mh::DistributedContext::set_current_world(world);
        return m;
    }();
    return mesh;
}

// Rank 1 runs in lockstep but reports nothing, so the ranks do not interleave output.
class NullReporter : public benchmark::BenchmarkReporter {
public:
    bool ReportContext(const Context&) override { return true; }
    void ReportRuns(const std::vector<Run>&) override {}
};

}  // namespace leg_bench
