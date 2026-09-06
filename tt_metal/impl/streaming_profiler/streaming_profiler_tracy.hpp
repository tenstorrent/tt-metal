// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <tracy/TracyTTDevice.hpp>
#include <tt-metalium/experimental/streaming_profiler.hpp>

#include "impl/streaming_profiler/streaming_profiler_consumer.hpp"

namespace tt::tt_metal::streaming_profiler {

class Service;

// The built-in Tracy sink: subscribes to every channel and pushes each record onto Tracy's device timeline, one
// context per (chip, core). Constructing it subscribes; destroying it unsubscribes. Everything after construction
// runs on the subscription's thread.
class TracySink {
public:
    explicit TracySink(Service& service);
    ~TracySink();
    TracySink(const TracySink&) = delete;
    TracySink& operator=(const TracySink&) = delete;

private:
    using Batch = experimental::streaming_profiler::Batch<experimental::streaming_profiler::Channel::All>;
    using Core = experimental::streaming_profiler::Core;
    struct Lane {
        TracyTTCtx ctx = nullptr;
        uint32_t thread = 0;
        uint32_t risc = 0;
    };
    // One Tracy context per core; each RISC's timeline row is created and named on first use.
    struct CoreEntry {
        TracyTTCtx ctx = nullptr;
        std::array<uint32_t, 5> thread{};
        uint8_t named = 0;
    };
    struct SrclocEntry {
        const char* name = nullptr;
        uint64_t key = 0;
        const void* srcloc = nullptr;
    };

    void on_batch(const Batch& batch);
    // Records are placed by their steady_clock time through a continuous piecewise-linear map onto Tracy's timeline:
    // a fresh segment per capture, then one per second whose slope is the two clocks' rate ratio measured over the
    // whole baseline since construction. Continuity keeps order and containment exact across segments.
    struct Probe {
        int64_t steady_ns;
        int64_t tracy_ns;
    };
    struct Segment {
        int64_t steady_ns;  // from here on
        int64_t tracy_ns;   // the map's value here
        double slope;       // timeline ns per steady_clock ns
    };
    Probe probe() const;
    double slope_since_base(const Probe& p) const;
    void start_capture_map();
    void refine_map();
    int64_t to_timeline(int64_t steady_ns) const;
    Lane lane(const Core& core);
    const void* srcloc(std::string_view name, uint32_t color, uint32_t risc);
    const void* srcloc_slow(std::string_view name, uint32_t color, uint32_t risc);
    void push_zone(const Core& core, std::string_view name, int64_t start_ns, int64_t end_ns, uint32_t color);
    void push_marker(
        const Core& core,
        std::string_view name,
        int64_t timestamp_ns,
        uint32_t runtime_id,
        std::span<const uint64_t> values);

    Service& service_;
    ConsumerHandle handle_ = 0;
    // The capture the map holds for, by its clock set; a new capture starts a new map.
    size_t capture_clocks_ = 0;
    int64_t capture_anchor_ = 0;
    int64_t anchor_tracy_ = 0;  // Tracy timer at construction; every context's cpuTime
    Probe base_{};              // taken at construction; every slope is measured against it
    std::vector<Segment> segments_;
    int64_t next_refine_ns_ = 0;
    uint64_t lane_key_ = ~uint64_t{0};
    Lane lane_hit_;
    std::unordered_map<uint64_t, CoreEntry> cores_;
    // Keyed by the name's address: a subscription's name strings never move or die while it lives.
    std::vector<SrclocEntry> srcloc_table_;  // open addressing, power-of-two size, at most half full
    size_t srcloc_count_ = 0;
    std::unordered_map<std::string, const void*> srclocs_;
};

}  // namespace tt::tt_metal::streaming_profiler
