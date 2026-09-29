// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Measurement-only instrumentation for #57586, enabled by TT_METAL_DISPATCH_STATS=1. Not for merging.
// Collects histograms of what the dispatch path and the host thread pools serve, and prints one
// `DISPATCH_STATS {json}` line to stderr when a command queue or a pool is destroyed.

#pragma once

#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace tt::tt_metal::dispatch_stats {

inline bool enabled() {
    static const bool on = [] {
        const char* e = std::getenv("TT_METAL_DISPATCH_STATS");
        return e != nullptr && e[0] != '\0' && e[0] != '0';
    }();
    return on;
}

inline uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// Values below 8 get bucket v. Above that, four buckets per power of two: bucket 4*e + m (e >= 3) holds
// [2^e * (4 + m) / 4, 2^e * (5 + m) / 4), so the resolution is about 19%.
struct Log2Hist {
    static constexpr size_t BUCKETS = 4 * 64;
    std::array<uint64_t, BUCKETS> buckets{};
    uint64_t count = 0;
    uint64_t sum = 0;
    uint64_t max = 0;

    static size_t bucket(uint64_t v) {
        if (v < 8) {
            return static_cast<size_t>(v);
        }
        const int e = std::bit_width(v) - 1;  // v in [2^e, 2^(e+1))
        const auto m = static_cast<size_t>((v >> (e - 2)) & 3);
        return (4 * static_cast<size_t>(e)) + m;
    }
    void add(uint64_t v) {
        buckets[bucket(v)]++;
        count++;
        sum += v;
        max = v > max ? v : max;
    }
    void merge(const Log2Hist& o) {
        for (size_t k = 0; k < BUCKETS; k++) {
            buckets[k] += o.buckets[k];
        }
        count += o.count;
        sum += o.sum;
        max = o.max > max ? o.max : max;
    }
    std::string json() const {
        std::string s = "{\"n\":" + std::to_string(count) + ",\"sum\":" + std::to_string(sum) +
                        ",\"max\":" + std::to_string(max) + ",\"q4\":{";
        bool first = true;
        for (size_t k = 0; k < BUCKETS; k++) {
            if (buckets[k] != 0) {
                s += (first ? "\"" : ",\"") + std::to_string(k) + "\":" + std::to_string(buckets[k]);
                first = false;
            }
        }
        return s + "}}";
    }
};

// Exact counts for small integers; larger values land in the last bucket.
struct CountHist {
    static constexpr size_t BUCKETS = 129;
    std::array<uint64_t, BUCKETS> buckets{};
    void add(uint64_t v) { buckets[v < BUCKETS ? v : BUCKETS - 1]++; }
    void merge(const CountHist& o) {
        for (size_t k = 0; k < BUCKETS; k++) {
            buckets[k] += o.buckets[k];
        }
    }
    std::string json() const {
        std::string s = "{";
        bool first = true;
        for (size_t k = 0; k < BUCKETS; k++) {
            if (buckets[k] != 0) {
                s += (first ? "\"" : ",\"") + std::to_string(k) + "\":" + std::to_string(buckets[k]);
                first = false;
            }
        }
        return s + "}";
    }
};

// Snapshots are cumulative. A process that exits without destroying its queues and pools still leaves data.
inline bool snapshot_due(uint64_t& last_ns) {
    constexpr uint64_t PERIOD_NS = 60'000'000'000ULL;
    const uint64_t now = now_ns();
    if (last_ns == 0) {
        last_ns = now;
        return false;
    }
    if (now - last_ns < PERIOD_NS) {
        return false;
    }
    last_ns = now;
    return true;
}

inline uint64_t next_instance_id() {
    static std::atomic<uint64_t> next{0};
    return next++;
}

inline void emit(const std::string& json) {
    std::fprintf(stderr, "DISPATCH_STATS %s\n", json.c_str());
    std::fflush(stderr);
    if (const char* path = std::getenv("TT_METAL_DISPATCH_STATS_FILE")) {
        if (FILE* f = std::fopen(path, "a")) {
            std::fprintf(f, "%s\n", json.c_str());
            std::fclose(f);
        }
    }
}

}  // namespace tt::tt_metal::dispatch_stats
