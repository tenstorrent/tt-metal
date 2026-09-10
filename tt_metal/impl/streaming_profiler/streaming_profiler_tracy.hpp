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

#include <string>
#include <unordered_set>

namespace tt::tt_metal::streaming_profiler {

class Service;

// The built-in Tracy sink: registers for every record type and pushes each record onto Tracy's device timeline, one
// context per (chip, core). Constructing it registers; destroying it unregisters. Everything after construction
// runs on the callback's thread.
class TracySink {
public:
    explicit TracySink(Service& service);
    ~TracySink();
    TracySink(const TracySink&) = delete;
    TracySink& operator=(const TracySink&) = delete;

private:
    using Batch = experimental::streaming_profiler::Batch<experimental::streaming_profiler::RecordType::All>;
    using Core = experimental::streaming_profiler::Core;
    struct Lane {
        TracyTTCtx ctx = nullptr;
        uint32_t thread = 0;
        uint32_t risc = 0;
    };
    // Each RISC's timeline row is created and named on first use.
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

    void on_batch(const Batch& batch, uint64_t capture);
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
    // Device<->device sync plots, all RATES. Per chip and per sync kind (the 3 us LOCAL tracker, the 1 ms LINK
    // stamps): the chip's applied AICLK over the ROOT chip's at the same instant -- the factor that scales its
    // wall-clock rate onto the root's; the root reads exactly 1. Each stream's AICLK comes from a sliding
    // dwall/drefclk over its PP_CLOCK samples. Plus the cross-chip refclk scale regression the d2d consumer publishes
    // through SyncPlots.
    struct FreqPoint {
        int64_t host_ns;
        double ghz;
    };
    std::vector<FreqPoint> compute_frequency(size_t begin, size_t end) const;
    const char* intern_name(const std::string& name);
    // Plots are emitted at capture end, not during decode: at decode time the correction is not yet solved
    // (lookup returns 0) and the timeline map has no segments (points land at raw, hours-off timestamps).
    void emit_plots();
    int64_t plot_stamp(int64_t host_ns) const;  // the PlotDataAt stamp for a point at this host time

    Service& service_;
    ConsumerHandle handle_ = 0;
    uint64_t capture_ = 0;      // the capture the map holds for; a new one starts a new map
    int64_t anchor_tracy_ = 0;  // Tracy timer at construction; every context's cpuTime
    Probe base_{};              // taken at construction; every slope is measured against it
    std::vector<Segment> segments_;
    int64_t next_refine_ns_ = 0;
    uint64_t lane_key_ = ~uint64_t{0};
    Lane lane_hit_;
    std::unordered_map<uint64_t, CoreEntry> cores_;
    // Keyed by the name's address: a callback's name strings never move or die while it lives.
    std::vector<SrclocEntry> srcloc_table_;  // open addressing, power-of-two size, at most half full
    size_t srcloc_count_ = 0;
    std::unordered_map<std::string, const void*> srclocs_;
    std::vector<DeviceClock> clocks_;  // per device index, for mapping a clock sample's device time to the timeline
    std::vector<DeviceClock> eth_clocks_;  // per device index, the idle-eth wall anchor for PP_CLOCK samples
    struct PlotSample {
        uint32_t dev;
        uint32_t kind;
        uint32_t core;  // eth core index on the device: one refclk counter per stream
        uint64_t ts;
        uint32_t value24;  // the 24-bit refclk reading
    };
    std::vector<PlotSample> plot_samples_;  // accumulated during the capture, drained in emit_plots()
    std::unordered_set<std::string> plot_names_;  // interned: PlotDataAt keys a plot by its name pointer
};

}  // namespace tt::tt_metal::streaming_profiler
