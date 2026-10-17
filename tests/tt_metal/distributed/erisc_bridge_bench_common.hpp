// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// One FILT surface and counter vocabulary for every leg: total_amt, warmup_pct, packet_size,
// verify, pace, sweep -- in that order. A leg may append its own axes, never insert. See below.
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>

namespace tt::tt_fabric::erisc_bridge::bench {

inline std::vector<std::string> common_arg_names() {
    return {"total_amt", "warmup_pct", "packet_size", "verify", "pace", "sweep"};
}

inline std::vector<std::string> arg_names_with(const std::vector<std::string>& leg_specific) {
    std::vector<std::string> n = common_arg_names();
    n.insert(n.end(), leg_specific.begin(), leg_specific.end());
    return n;
}

// Fixed positions, so state.range(i) means the same thing in every leg.
enum ArgIndex : int {
    kTotalAmt = 0,
    kWarmupPct = 1,
    kPacketSize = 2,
    kVerify = 3,
    kPace = 4,
    kSweep = 5,
    kLegSpecific0 = 6,
};

inline constexpr std::uint64_t kMiB = 1024ull * 1024ull;

// The payload pattern: word 0 the frame index, 1-2 a device stamp, word 3 on this tag. Here, not
// with the producer, because the H2H benchmark checks it and has no device. The T6 kernel agrees.
inline constexpr std::uint32_t kPayloadTag = 0xC0DE0000u;
inline constexpr std::uint32_t kPayloadFirstTagWord = 3;
constexpr std::uint32_t payload_tag_word(std::uint32_t k) { return kPayloadTag | (k & 0xFFFFu); }

// Linear interpolation. sorted[(n*99)/100] equals n-1 below 100 samples, which would collapse
// p99 into max.
template <typename T>
inline double pct_of(const std::vector<T>& sorted, double p) {
    if (sorted.empty()) {
        return 0.0;
    }
    const double x = p * (static_cast<double>(sorted.size()) - 1.0);
    const size_t lo = static_cast<size_t>(x);
    const size_t hi = lo + 1 < sorted.size() ? lo + 1 : lo;
    return static_cast<double>(sorted[lo]) +
           (x - static_cast<double>(lo)) * (static_cast<double>(sorted[hi]) - static_cast<double>(sorted[lo]));
}

// Burst and sustained are different measurements and must not share a name: while frames <= ring
// depth the sender never feels backpressure, so a slow drain would not show.
inline void emit_throughput(benchmark::State& state, double mbps, std::uint64_t frames, bool sustained) {
    state.counters["throughput_MBps"] = mbps;
    state.counters["throughput_GBps"] = mbps / 1000.0;
    state.counters["frames"] = static_cast<double>(frames);
    state.counters["sustained"] = sustained ? 1.0 : 0.0;
}

// A DECLARED ARG THE LEG IGNORES IS WORSE THAN NO ARG: a sweep=1 run would report a clean pass
// while covering one link.
inline bool sweep_requested(const benchmark::State& state) { return state.range(kSweep) != 0; }
inline void sweep_not_implemented(benchmark::State& state) {
    state.SkipWithError("sweep=1 requested, but this leg still covers a single configuration");
}

// combos_passed < combos is a FAILURE even when throughput looks healthy -- the combinations
// that delivered nothing contributed no frames and so did not drag the average down.
inline void emit_sweep(benchmark::State& state, std::uint32_t combos, std::uint32_t passed) {
    state.counters["combos"] = static_cast<double>(combos);
    state.counters["combos_passed"] = static_cast<double>(passed);
    if (passed != combos) {
        state.SkipWithError("sweep: not every (link, channel) combination delivered");
    }
}

// ONE-WAY, and only valid where ONE clock stamps both ends -- E2H (release -> arrival) and H2E
// (inject -> consumed). Those two spans mean different things and are not comparable.
inline void emit_latency_oneway(benchmark::State& state, std::vector<double> us, double bias_us = 0.0) {
    if (us.empty()) {
        return;
    }
    std::sort(us.begin(), us.end());
    state.counters["latency_us_min"] = us.front();
    state.counters["latency_us_p50"] = pct_of(us, 0.50);
    state.counters["latency_us_p99"] = pct_of(us, 0.99);
    state.counters["latency_us_max"] = us.back();
    state.counters["latency_us_bias"] = bias_us;
    state.counters["latency_samples"] = static_cast<double>(us.size());
}

// Round trip, for a span whose ends are stamped by different clocks. Never reported as
// latency_us_*: half an RTT is not the one-way time, and the asymmetry is unknown.
inline void emit_rtt_us(benchmark::State& state, std::vector<double> us) {
    if (us.empty()) {
        return;
    }
    std::sort(us.begin(), us.end());
    state.counters["rtt_us_min"] = us.front();
    state.counters["rtt_us_p50"] = pct_of(us, 0.50);
    state.counters["rtt_us_p99"] = pct_of(us, 0.99);
    state.counters["rtt_us_max"] = us.back();
    state.counters["rtt_samples"] = static_cast<double>(us.size());
}

inline void emit_latency_us(benchmark::State& state, std::vector<double> us, double uncertainty_us) {
    if (us.empty()) {
        return;
    }
    std::sort(us.begin(), us.end());
    state.counters["latency_us_min"] = us.front();
    state.counters["latency_us_p50"] = pct_of(us, 0.50);
    state.counters["latency_us_p99"] = pct_of(us, 0.99);
    state.counters["latency_us_max"] = us.back();
    state.counters["latency_us_uncertainty"] = uncertainty_us;
    state.counters["latency_samples"] = static_cast<double>(us.size());
}

// DEVICE COST IN CYCLES, NEVER CONVERTED. AICLK scales, so a fixed cycles_per_us is wrong and
// varying; these stay comparable against themselves and nothing else.
inline void emit_device_cycles(
    benchmark::State& state, std::vector<std::uint32_t> issue, std::vector<std::uint32_t> stall) {
    if (!issue.empty()) {
        std::sort(issue.begin(), issue.end());
        state.counters["issue_cyc_p50"] = pct_of(issue, 0.50);
        state.counters["issue_cyc_p99"] = pct_of(issue, 0.99);
    }
    if (!stall.empty()) {
        std::sort(stall.begin(), stall.end());
        state.counters["stall_cyc_p50"] = pct_of(stall, 0.50);
        state.counters["stall_cyc_p99"] = pct_of(stall, 0.99);
    }
}

// CORRECTNESS, every leg. A perf number from a broken stream is worse than no number.
inline void emit_health(
    benchmark::State& state, std::uint64_t verify_fail, std::uint64_t credit_stalls, std::uint64_t bad_order = 0) {
    state.counters["verify_fail"] = static_cast<double>(verify_fail);
    state.counters["credit_stalls"] = static_cast<double>(credit_stalls);
    state.counters["bad_order"] = static_cast<double>(bad_order);
}

}  // namespace tt::tt_fabric::erisc_bridge::bench
